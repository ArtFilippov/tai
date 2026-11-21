"""Doctor behavior through independent conversations and durable outcomes."""

import hashlib
import json
import random
import subprocess
import sys
import unittest
from collections.abc import Sequence
from pathlib import Path

import pytest

from attack.assessed_candidates import AssessedCandidates
from attack.attack_run import AttackRun
from attack.doctor.doctor_probe import DoctorProbe
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
from .doctor_reference_utils import reference


class TestDoctorProbe(unittest.TestCase):
    """Preserve upstream candidates while assessing original-request fulfillment."""

    def attempt(self, path: Path, text: str, target: Chat, *, judge: Judge,
                techniques: Sequence[str] = ("puppetry", "bypass", "bypass_leet"), seed: int = 0) -> str | None:
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
            probe=DoctorProbe(
                base,
                AssessedCandidates(journal),
                journal, techniques=techniques, seed=seed
            )
        ).run()

    def test_all_candidates_match_pinned_source_with_exact_multiline_request(self) -> None:
        base = "  Объясни café с {двумя} примерами\nи ссылками \\1  "
        for seed in (0, 41):
            with temporary_journal("parity.jsonl") as path:
                target = ScriptedChat(["Negative"] * 5)
                self.attempt(
                    path, base, target,
                    judge=EqualJudge("Complete"),
                    seed=seed
                )
                assert [hashlib.sha256(target.transcript(index).turns[0].content.text.encode()).hexdigest() for index in range(5)] == [hashlib.sha256(text.encode()).hexdigest() for text in reference(base, seed)], "Doctor changed upstream templates, substitution boundaries, suffix or encoding"

    def test_reversed_and_duplicate_selection_keeps_canonical_order_and_fresh_histories(self) -> None:
        with temporary_journal("canonical.jsonl") as path:
            target = ScriptedChat(["Negative"] * 5)
            self.attempt(
                path, "Base", target, judge=EqualJudge("Complete"),
                techniques=["bypass_leet", "bypass", "puppetry", "bypass"]
            )
            events = [event for event in ParsedJournal(path).events() if event.get("action") == "doctor_transformation"]
            assert ([event["candidate"] for event in events], [len(target.transcript(index).turns) for index in range(5)]) == (
                [
                    {
                        "technique": "puppetry",
                        "template": 1,
                        "position": 1
                    },
                    {
                        "technique": "bypass",
                        "template": 1,
                        "position": 2
                    },
                    {
                        "technique": "bypass",
                        "template": 2,
                        "position": 3
                    },
                    {
                        "technique": "bypass_leet",
                        "template": 1,
                        "position": 4,
                        "seed": 0
                    },
                    {
                        "technique": "bypass_leet",
                        "template": 2,
                        "position": 5,
                        "seed": 0
                    }
                ],
                [1] * 5
            ), "Selection reordered or duplicated candidates, or Target histories were shared"

    def test_first_success_does_not_transform_later_templates(self) -> None:
        with temporary_journal("early.jsonl") as path:
            result = self.attempt(
                path, "Base",
                ScriptedChat(["Complete"]),
                judge=EqualJudge("Complete")
            )
            outcome = ParsedJournal(path).read()
            assert (result, [event.get("action", event["event"]) for event in ParsedJournal(path).events()], outcome["successful_candidate"], outcome["calls"]) == (
                "Complete",
                [
                    "start", "doctor_transformation", "call", "result", "call", "judgment", "doctor_success", "end"
                ],
                {
                    "technique": "puppetry",
                    "template": 1,
                    "position": 1
                },
                {
                    "target": 1,
                    "attacker": 0,
                    "judge": 1
                }
            ), "Success executed later candidates or lost result-before-assessment ordering"

    def test_exhaustion_retains_the_last_assessed_response_including_empty_text(self) -> None:
        for last in ("  Last\n  ", ""):
            with temporary_journal("exhausted.jsonl") as path:
                result = self.attempt(
                    path, "Base",
                    ScriptedChat(["Negative"] * 4 + [last]),
                    judge=EqualJudge("Complete")
                )
                outcome = ParsedJournal(path).read()
                assert (
                    result, outcome["text"], outcome["status"], outcome["reason"], outcome["assessment"],
                    outcome.get("successful_candidate"),
                    outcome["calls"]
                ) == (
                    last, last, "round_limit", "candidates_exhausted", False, None,
                    {
                        "target": 5,
                        "attacker": 0,
                        "judge": 5
                    }
                ), "Exhaustion lost the last verdict, invented success or confused empty text with no result"

    def test_leet_judge_receives_original_request_and_full_raw_response(self) -> None:
        with temporary_journal("judge.jsonl") as path:
            base = "  Explain café\nwith two examples  "
            answer = "  H34d3r\nFull answer\nTrailing text  "
            chat = ScriptedChat(["YES"])
            result = self.attempt(
                path, base,
                ScriptedChat([answer]),
                judge=QwenJudge(
                    base, chat
                ),
                techniques=["bypass_leet"], seed=17
            )
            assert (
                result,
                json.loads(
                    chat.transcript(0).turns[-1].content.text
                ),
                ParsedJournal(path).read()["successful_candidate"]
            ) == (
                answer,
                {
                    "base_request": base,
                    "presented_result": answer
                },
                {
                    "technique": "bypass_leet",
                    "template": 1,
                    "position": 1,
                    "seed": 17
                }
            ), "Judge received a transformed request or decoded, extracted or truncated response"

    def test_leet_repeats_independently_of_other_techniques_without_global_random_mutation(self) -> None:
        state = random.getstate()
        prompts: list[list[str]] = []
        for selection in (["bypass_leet"], ["puppetry", "bypass", "bypass_leet"], ["bypass_leet"]):
            with temporary_journal("random.jsonl") as path:
                self.attempt(
                    path, "Explain café",
                    ScriptedChat(["Negative"] * 5),
                    judge=EqualJudge("Complete"),
                    techniques=selection, seed=29
                )
                events = list(
                    ParsedJournal(path).events()
                )
                texts = [str(event["prompt"]) for event in events if event.get("action") == "doctor_transformation"]
                prompts.append(texts[-2:])
        assert prompts[0] == prompts[1] == prompts[2] and random.getstate() == state, "Leetspeak depends on other techniques, previous runs or process-global randomness"

    def test_journal_preserves_original_provenance_source_settings_and_candidate(self) -> None:
        with temporary_journal("metadata.jsonl") as path:
            self.attempt(
                path, "  Base café\n  ",
                ScriptedChat(["Complete"]),
                judge=EqualJudge("Complete"),
                techniques=["bypass_leet"], seed=7
            )
            events = list(
                ParsedJournal(path).events()
            )
            probe = json.loads(
                path.read_text(encoding="utf-8")
                .splitlines()[0]
            )["attack"]["probe"]
            assert (events[0]["base_metadata"], events[1]["original"], events[1]["base_metadata"], events[2]["metadata"], probe["source"], probe["revision"], probe["techniques"], probe["seed"]) == (
                {
                    "id": 23,
                    "microcategory": "specific",
                    "macro": "general",
                    "source_category": "original",
                    "source_revision": "base-revision",
                    "source_sha256": "base-digest",
                    "taxonomy_version": "1",
                    "taxonomy_sha256": "taxonomy-digest"
                },
                "  Base café\n  ", events[0]["base_metadata"],
                {
                    "candidate": {
                        "technique": "bypass_leet",
                        "template": 1,
                        "position": 1,
                        "seed": 7
                    }
                },
                "garak.probes.doctor", "3f50ea5ff9cd7050099940647c15c39b07a93392",
                ["bypass_leet"],
                7
            ), "Journal lost original provenance, source revision, effective selection or candidate settings"

    def test_target_error_after_negative_reply_aborts_without_fallback(self) -> None:
        with temporary_journal("target-error.jsonl") as path:
            with pytest.raises(
                ValueError, match="Target unavailable"
            ):
                self.attempt(
                    path, "Base",
                    InterruptedChat(
                        ["Negative"], "Target unavailable"
                    ),
                    judge=EqualJudge("Complete")
                )
            outcome = ParsedJournal(path).read()
            assert (outcome["status"], outcome["reason"], outcome["text"], outcome["assessment"], outcome["calls"]) == (
                "failure", "execution_error", "Negative", False,
                {
                    "target": 1,
                    "attacker": 0,
                    "judge": 1
                }
            ), "A Target error was retried, skipped, replaced with exhaustion or lost the previous assessment"

    def test_malformed_judge_verdict_aborts_without_assessing_later_candidates(self) -> None:
        with temporary_journal("judge-error.jsonl") as path:
            with pytest.raises(
                ValueError, match="Malformed judge verdict"
            ):
                self.attempt(
                    path, "Base",
                    ScriptedChat(["Selected"]),
                    judge=QwenJudge(
                        "Base",
                        ScriptedChat(["invalid"])
                    )
                )
            outcome = ParsedJournal(path).read()
            assert (
                outcome["status"], outcome["reason"], outcome["text"], outcome["assessment"], outcome["calls"],
                outcome.get("successful_candidate")
            ) == (
                "failure", "assessment_error", "Selected", None,
                {
                    "target": 1,
                    "attacker": 0,
                    "judge": 1
                },
                None
            ), "Judge failure became a negative verdict, erased the result or executed another candidate"

    def test_fresh_process_recovers_the_recorded_winner_without_participants(self) -> None:
        with temporary_journal("recover.jsonl") as path:
            self.attempt(
                path, "Base",
                ScriptedChat(["Negative", "Complete"]),
                judge=EqualJudge("Complete"),
                techniques=["bypass_leet"], seed=11
            )
            process = subprocess.run(
                [
                    sys.executable, "-c", "import json, sys; from pathlib import Path; from journal.parsed_journal import ParsedJournal; print(json.dumps(ParsedJournal(Path(sys.argv[1])).read()))",
                    str(path)
                ],
                check=True, capture_output=True, text=True
            )
            assert json.loads(process.stdout)["successful_candidate"] == {
                "technique": "bypass_leet",
                "template": 2,
                "position": 2,
                "seed": 11
            }, "Fresh-process recovery lost the recorded winning template or seed"
