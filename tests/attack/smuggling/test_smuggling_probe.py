"""Smuggling behavior through the public attack and durable journal boundaries."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from attack.assessed_candidates import AssessedCandidates
from attack.attack_run import AttackRun
from attack.smuggling.smuggling_probe import SmugglingProbe
import pytest

from base.alert_base import AlertBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.equal_judge import EqualJudge
from judge.judge import Judge
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.chat import Chat
from target.interrupted_chat import InterruptedChat
from target.scripted_chat import ScriptedChat
from ...journal_utils import temporary_journal


class TestSmugglingProbe(unittest.TestCase):
    """Transform only the supplied request and assess one complete Target reply."""

    def attempt(self, path: Path, text: str, target: Chat, *,
                judge: Judge) -> str | None:
        journal = VersionedJournal(
            DiskJournal(path)
        )
        base = AlertBase(
            23, text, "specific",
            {
                "macro": "general",
                "source_category": "original",
                "source_revision": "base-revision",
                "source_sha256": "base-digest",
                "taxonomy_version": "1",
                "taxonomy_sha256": "taxonomy-digest"
            }
        )
        return AttackRun(
            base, target,
            RecordedJudge(
                judge, journal, text
            ),
            journal,
            probe=SmugglingProbe(
                base,
                AssessedCandidates(journal),
                journal
            )
        ).run()

    def test_one_deterministic_candidate_uses_an_independent_exchange(self) -> None:
        with temporary_journal("smuggling.jsonl") as path:
            target = ScriptedChat(["Complete answer"])
            self.attempt(
                path, "a c e i o p s x y A B C E H K M O P T X!", target,
                judge=EqualJudge("Complete answer")
            )
            assert [(turn.role, turn.content.text) for turn in target.transcript(0).turns] == [
                ("user", "а с е ı о р ѕ х у А В С Е Н К М О Р Т Х!")
            ], "Smuggling changed the native substitutions or sent more than the caller's transformed request"

    def test_success_recovers_technique_and_exact_response(self) -> None:
        with temporary_journal("winner.jsonl") as path:
            result = self.attempt(
                path, "Base request",
                ScriptedChat(["Complete answer"]),
                judge=EqualJudge("Complete answer")
            )
            outcome = ParsedJournal(path).read()
            assert (result, outcome["assessment"], outcome["successful_candidate"], outcome["calls"]) == (
                "Complete answer", True,
                {
                    "technique": "homoglyph",
                    "position": 1,
                    "seed": 0
                },
                {
                    "target": 1,
                    "attacker": 0,
                    "judge": 1
                }
            ), "Success lost its technique, full response, verdict, or one-candidate call budget"

    def test_transformation_matches_pinned_upstream_and_repeats_from_seed_zero(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ\niiiiiiiiiiiiii café аіı 🙂 0123"
        reference = subprocess.run(
            [
                sys.executable, "-c", "import json, sys; from garak.probes.smuggling import _homoglyph_replace, DEFAULT_HOMOGLYPH_MAP; " "print(json.dumps(_homoglyph_replace(sys.argv[1], DEFAULT_HOMOGLYPH_MAP, seed=0)))", text
            ],
            check=True, capture_output=True, text=True
        )
        prompts: list[str] = []
        for index in range(2):
            with temporary_journal(f"native-{index}.jsonl") as path:
                target = ScriptedChat(["Complete answer"])
                self.attempt(
                    path, text, target,
                    judge=EqualJudge("Complete answer")
                )
                prompts.append(
                    target.transcript(0).turns[0].content.text
                )
        assert prompts == [
            json.loads(reference.stdout)
        ] * 2, "Transformation diverged from pinned garak or reused random state across runs"

    def test_fulfillment_judge_receives_original_request_and_complete_response(self) -> None:
        with temporary_journal("full.jsonl") as path:
            chat = ScriptedChat(["YES"])
            answer = "  Header\nFull café answer.\nTrailing text  "
            result = self.attempt(
                path, "  Base café\nexactly  ",
                ScriptedChat([answer]),
                judge=QwenJudge(
                    "  Base café\nexactly  ", chat
                )
            )
            assert (
                result,
                json.loads(
                    chat.transcript(0).turns[-1].content.text
                )
            ) == (
                answer,
                {
                    "base_request": "  Base café\nexactly  ",
                    "presented_result": answer
                }
            ), "The judge assessed transformed text or an extracted or rewritten response"

    def test_exhaustion_retains_negative_assessment_even_for_empty_response(self) -> None:
        for answer in ("No", ""):
            with self.subTest(answer=answer), temporary_journal("exhausted.jsonl") as path:
                result = self.attempt(
                    path, "Base",
                    ScriptedChat([answer]),
                    judge=EqualJudge("Complete answer")
                )
                outcome = ParsedJournal(path).read()
                assert (
                    result, outcome["status"], outcome["reason"], outcome["assessment"],
                    outcome.get("successful_candidate")
                ) == (
                    answer, "round_limit", "candidates_exhausted", False, None
                ), "Exhaustion lost the assessed response, invented a winner, or confused empty text with no result"

    def test_unchanged_text_skips_target_and_judge_without_fallback(self) -> None:
        with temporary_journal("skip.jsonl") as path:
            result = self.attempt(
                path, "123 !\nЖ🙂",
                ScriptedChat([]),
                judge=QwenJudge(
                    "123 !\nЖ🙂",
                    ScriptedChat([])
                )
            )
            outcome = ParsedJournal(path).read()
            events = list(
                ParsedJournal(path).events()
            )
            assert (result, outcome["status"], outcome["reason"], outcome["assessment"], outcome["calls"], events[1]["reason"]) == (
                None, "no_result", "no_candidates", None,
                {
                    "target": 0,
                    "attacker": 0,
                    "judge": 0
                },
                "unchanged_text"
            ), "Unchanged input was sent or assessed, lost its skip reason, or became a negative result"

    def test_journal_preserves_source_settings_original_provenance_and_candidate(self) -> None:
        with temporary_journal("metadata.jsonl") as path:
            self.attempt(
                path, "  Base café\nexactly  ",
                ScriptedChat(["Complete answer"]),
                judge=EqualJudge("Complete answer")
            )
            events = list(
                ParsedJournal(path).events()
            )
            metadata = {
                "id": 23,
                "microcategory": "specific",
                "macro": "general",
                "source_category": "original",
                "source_revision": "base-revision",
                "source_sha256": "base-digest",
                "taxonomy_version": "1",
                "taxonomy_sha256": "taxonomy-digest"
            }
            assert (
                events[0]["base"], events[0]["base_metadata"],
                json.loads(
                    path.read_text(encoding="utf-8")
                    .splitlines()[0]
                )["attack"]["probe"],
                events[1]["original"], events[1]["base_metadata"], events[2]["metadata"]
            ) == (
                "  Base café\nexactly  ", metadata,
                {
                    "type": "smuggling_probe",
                    "techniques": ["homoglyph"],
                    "source": "garak.probes.smuggling._homoglyph_replace",
                    "revision": "3f50ea5ff9cd7050099940647c15c39b07a93392",
                    "settings": {
                        "seed": 0,
                        "candidates": 1,
                        "replacement": "every eligible character",
                        "mapping": {
                            "a": ["а"],
                            "c": ["с"],
                            "e": ["е"],
                            "i": [
                                "і", "ı"
                            ],
                            "o": ["о"],
                            "p": ["р"],
                            "s": ["ѕ"],
                            "x": ["х"],
                            "y": ["у"],
                            "A": ["А"],
                            "B": ["В"],
                            "C": ["С"],
                            "E": ["Е"],
                            "H": ["Н"],
                            "K": ["К"],
                            "M": ["М"],
                            "O": ["О"],
                            "P": ["Р"],
                            "T": ["Т"],
                            "X": ["Х"]
                        }
                    }
                },
                "  Base café\nexactly  ", metadata,
                {
                    "candidate": {
                        "technique": "homoglyph",
                        "position": 1,
                        "seed": 0
                    }
                }
            ), "The journal lost source revision, effective mapping/settings, original provenance, or distinct candidate identity"

    def test_presented_result_precedes_assessment_and_success(self) -> None:
        with temporary_journal("order.jsonl") as path:
            self.attempt(
                path, "Base",
                ScriptedChat(["Complete answer"]),
                judge=EqualJudge("Complete answer")
            )
            assert [event.get("action", event["event"]) for event in ParsedJournal(path).events()] == [
                "start", "smuggling_transformation", "call", "result", "call", "judgment", "smuggling_success", "end"
            ], "The result was not saved before assessment or success preceded the verdict"

    def test_target_failure_remains_a_failed_run(self) -> None:
        with temporary_journal("target-error.jsonl") as path:
            with pytest.raises(
                ValueError, match="Target unavailable"
            ):
                self.attempt(
                    path, "Base",
                    InterruptedChat(
                        [], "Target unavailable"
                    ),
                    judge=EqualJudge("Complete answer")
                )
            outcome = ParsedJournal(path).read()
            assert (outcome["status"], outcome["reason"], outcome["assessment"], outcome["text"]) == (
                "failure", "execution_error", None, None
            ), "A Target error became negative assessment, exhaustion, or a fabricated response"

    def test_judge_failure_keeps_presented_result_without_negative_verdict(self) -> None:
        with temporary_journal("judge-error.jsonl") as path:
            with pytest.raises(
                ValueError, match="Malformed judge verdict"
            ):
                self.attempt(
                    path, "Base",
                    ScriptedChat(["Selected response"]),
                    judge=QwenJudge(
                        "Base",
                        ScriptedChat(["invalid"])
                    )
                )
            outcome = ParsedJournal(path).read()
            assert (
                outcome["status"], outcome["reason"], outcome["assessment"], outcome["text"],
                outcome.get("successful_candidate")
            ) == (
                "failure", "assessment_error", None, "Selected response", None
            ), "A Judge error erased the selected result, became a negative verdict, or invented a winner"

    def test_fresh_process_recovers_success_without_participants(self) -> None:
        with temporary_journal("recover.jsonl") as path:
            self.attempt(
                path, "Base",
                ScriptedChat(["Complete answer"]),
                judge=EqualJudge("Complete answer")
            )
            process = subprocess.run(
                [
                    sys.executable, "-c", "import json, sys; from pathlib import Path; from journal.parsed_journal import ParsedJournal; " "print(json.dumps(ParsedJournal(Path(sys.argv[1])).read()))",
                    str(path)
                ],
                check=True, capture_output=True, text=True
            )
            assert json.loads(process.stdout)["successful_candidate"] == {
                "technique": "homoglyph",
                "position": 1,
                "seed": 0
            }, "A fresh process failed to recover the recorded successful technique"
