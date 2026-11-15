"""DAN behavior through the public text-returning attack boundary."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import pytest

from attack.attack_run import AttackRun
from attack.dan.dan_probe import DanProbe
from base.alert_base import AlertBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.equal_judge import EqualJudge
from judge.judge import Judge
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from prompt_collection.prompt_collection import PromptCollection
from prompt_collection.provided_prompt import ProvidedPrompt
from prompt_collection.scripted_prompts import ScriptedPrompts
from target.chat import Chat
from target.interrupted_chat import InterruptedChat
from target.scripted_chat import ScriptedChat
from ...journal_utils import temporary_journal


class TestDanProbe(unittest.TestCase):
    """Keep activation diagnostic and judge only the exact request's full reply."""

    def prompts(self, *, broken: bool = False) -> PromptCollection:
        return ScriptedPrompts(
            [
                ProvidedPrompt(
                    "Activate",
                    {
                        "section": "fake",
                        "variant": "Original",
                        "source_index": 41
                    }
                ),
                ProvidedPrompt(
                    "Activate",
                    {
                        "section": "fake",
                        "variant": "Original",
                        "source_index": 7
                    }
                )
            ],
            broken=broken
        )

    def attempt(self, path: Path, target: Chat, *, judge: Judge,
                prompts: PromptCollection) -> str | None:
        journal = VersionedJournal(
            DiskJournal(path)
        )
        base = AlertBase(
            23, "  Explain café\nexactly.  ", "specific",
            {
                "macro": "general",
                "source_category": "original-category",
                "source_revision": "pinned-revision",
                "source_sha256": "original-digest",
                "taxonomy_version": "1",
                "taxonomy_sha256": "taxonomy-digest"
            }
        )
        return AttackRun(
            base, target,
            RecordedJudge(
                judge, journal,
                base.text()
            ),
            journal,
            probe=DanProbe(
                base, prompts, journal
            )
        ).run()

    def test_candidates_have_independent_two_exchange_histories(self) -> None:
        with temporary_journal("dan-histories.jsonl") as path:
            target = ScriptedChat(["I refuse activation", "No", "Persona enabled", "Still no"])
            self.attempt(
                path, target,
                judge=EqualJudge("Complete answer"),
                prompts=self.prompts()
            )
            assert [
                [(turn.role, turn.content.text) for turn in target.transcript(index).turns]
                for index in range(4)
            ] == [
                [("user", "Activate")],
                [("user", "Activate"), ("assistant", "I refuse activation"), ("user", "  Explain café\nexactly.  ")],
                [("user", "Activate")],
                [("user", "Activate"), ("assistant", "Persona enabled"), ("user", "  Explain café\nexactly.  ")],
            ], "Candidates shared history, lost actual activation replies, or changed the base request"

    def test_first_success_recovers_source_identity_and_configured_target(self) -> None:
        with temporary_journal("dan-winner.jsonl") as path:
            target = ScriptedChat(["No persona acknowledgment", "Complete answer"])
            result = self.attempt(
                path, target,
                judge=EqualJudge("Complete answer"),
                prompts=self.prompts(broken=True)
            )
            outcome = ParsedJournal(path).read()
            assert (
                result, outcome["successful_candidate"], outcome["target"], outcome["calls"],
            ) == (
                "Complete answer",
                {
                    "section": "fake",
                    "variant": "Original",
                    "source_index": 41,
                    "position": 1
                },
                {
                    "type": "scripted_chat",
                    "replies": [
                        "No persona acknowledgment", "Complete answer"
                    ]
                },
                {
                    "target": 2,
                    "attacker": 0,
                    "judge": 1
                }
            ), "Success lost its source identity or Target, or continued collection traversal"

    def test_second_success_distinguishes_duplicate_prompts_by_source(self) -> None:
        with temporary_journal("dan-second.jsonl") as path:
            self.attempt(
                path,
                ScriptedChat(["Acknowledged", "No", "Refused activation", "Complete answer"]),
                judge=EqualJudge("Complete answer"),
                prompts=self.prompts(broken=True)
            )
            assert ParsedJournal(path).read()["successful_candidate"] == {
                "section": "fake",
                "variant": "Original",
                "source_index": 7,
                "position": 2
            }, "The winner was inferred from duplicate text or traversal position instead of its source"

    def test_exhaustion_retains_last_assessment_without_a_winner(self) -> None:
        with temporary_journal("dan-exhausted.jsonl") as path:
            result = self.attempt(
                path,
                ScriptedChat(["Persona enabled", "No", "Persona enabled", "Last answer"]),
                judge=EqualJudge("Complete answer"),
                prompts=self.prompts()
            )
            outcome = ParsedJournal(path).read()
            assert (
                result, outcome["status"], outcome["reason"], outcome["assessment"], "successful_candidate" in outcome,
            ) == (
                "Last answer", "round_limit", "candidates_exhausted", False, False,
            ), "Persona acknowledgment became success or exhaustion lost its last assessed response"

    def test_judge_receives_complete_second_reply_and_original_request(self) -> None:
        with temporary_journal("dan-full.jsonl") as path:
            chat = ScriptedChat(["YES"])
            result = self.attempt(
                path,
                ScriptedChat(["Activation only", "[GPT] No.\n[DAN] Complete answer.\nTrailing text"]),
                judge=QwenJudge(
                    "  Explain café\nexactly.  ", chat
                ),
                prompts=self.prompts()
            )
            assert (
                result,
                json.loads(
                    chat.transcript(0).turns[-1].content.text
                )
            ) == (
                "[GPT] No.\n[DAN] Complete answer.\nTrailing text",
                {
                    "base_request": "  Explain café\nexactly.  ",
                    "presented_result": "[GPT] No.\n[DAN] Complete answer.\nTrailing text"
                }
            ), "Assessment lost the exact base request, extracted a subsection, or included activation text"

    def test_saved_exchanges_preserve_phases_actual_replies_and_source(self) -> None:
        with temporary_journal("dan-phases.jsonl") as path:
            target = ScriptedChat(["Actual activation", "Complete answer"])
            self.attempt(
                path, target,
                judge=EqualJudge("Complete answer"),
                prompts=self.prompts()
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert [(event["phase"], event["response"], event["candidate"]) for event in events if event.get("role") == "target"] == [
                    (
                        "activation", "Actual activation",
                        {
                            "section": "fake",
                            "variant": "Original",
                            "source_index": 41,
                            "position": 1
                        }
                    ),
                    (
                        "base_request", "Complete answer",
                        {
                            "section": "fake",
                            "variant": "Original",
                            "source_index": 41,
                            "position": 1
                        }
                    )
                ], "The journal lost exchange phases, actual replies, or stable source identity"

    def test_second_message_preserves_original_request_provenance(self) -> None:
        with temporary_journal("dan-provenance.jsonl") as path:
            target = ScriptedChat(["Actual activation", "Complete answer"])
            self.attempt(
                path, target,
                judge=EqualJudge("Complete answer"),
                prompts=self.prompts()
            )
            assert target.transcript(1).turns[-1].content.notes == {
                "id": 23,
                "source_category": "original-category",
                "microcategory": "specific",
                "macro": "general",
                "source_revision": "pinned-revision",
                "source_sha256": "original-digest",
                "taxonomy_version": "1",
                "taxonomy_sha256": "taxonomy-digest"
            }, "The base request lost its original provenance in the second exchange"

    def test_candidate_and_presented_result_are_saved_before_judgment(self) -> None:
        with temporary_journal("dan-selection.jsonl") as path:
            self.attempt(
                path,
                ScriptedChat(["Actual activation", "Complete answer"]),
                judge=EqualJudge("Complete answer"),
                prompts=self.prompts()
            )
            assert [event.get("action", event["event"]) for event in ParsedJournal(path).events()] == [
                    "start", "call", "call", "dan_candidate", "result", "call", "judgment", "dan_success", "end"
                ], "The candidate or result was recorded after judgment, or success preceded the verdict"

    def test_empty_collection_preserves_no_result_contract(self) -> None:
        with temporary_journal("dan-empty.jsonl") as path:
            result = self.attempt(
                path,
                ScriptedChat([]),
                judge=EqualJudge("Complete answer"),
                prompts=ScriptedPrompts([])
            )
            outcome = ParsedJournal(path).read()
            assert (result, outcome["status"], outcome["reason"]) == (
                None, "no_result", "no_candidates",
            ), "An empty collection fabricated a response or a successful candidate"

    def test_collection_accepts_source_identity_without_dan_fields(self) -> None:
        with temporary_journal("dan-generic-source.jsonl") as path:
            self.attempt(
                path,
                ScriptedChat(["Actual activation", "Complete answer"]),
                judge=EqualJudge("Complete answer"),
                prompts=ScriptedPrompts(
                    [
                        ProvidedPrompt(
                            "Generic prompt",
                            {
                                "dataset": "examples",
                                "record": 23
                            }
                        )
                    ]
                )
            )
            assert ParsedJournal(path).read()["successful_candidate"] == {
                "dataset": "examples",
                "record": 23,
                "position": 1
            }, "A generic source required DAN-specific fields or lost its original identity"

    def test_target_failures_retain_completed_exchanges_without_exhaustion(self) -> None:
        for replies in ([], ["Activation reply"], ["Activation reply", "No"]):
            with self.subTest(replies=replies), temporary_journal("dan-target-error.jsonl") as path:
                completed = len(replies)
                with pytest.raises(
                    ValueError, match="Target unavailable"
                ):
                    self.attempt(
                        path,
                        InterruptedChat(
                            replies, "Target unavailable"
                        ),
                        judge=EqualJudge("Complete answer"),
                        prompts=self.prompts()
                    )
                outcome = ParsedJournal(path).read()
                assert (
                    outcome["calls"], outcome["status"], outcome["reason"], outcome["text"], "successful_candidate" in outcome,
                ) == (
                    {
                        "target": completed,
                        "attacker": 0,
                        "judge": completed // 2
                    },
                    "failure", "execution_error", "No" if completed == 2 else None, False
                ), "A Target failure skipped a candidate, erased completed exchanges, or became exhaustion"

    def test_context_failure_does_not_shorten_the_activation_or_base_request(self) -> None:
        with temporary_journal("dan-context-error.jsonl") as path:
            with pytest.raises(
                ValueError, match="Context size exceeded"
            ):
                self.attempt(
                    path,
                    InterruptedChat(
                        ["Activation reply"], "Context size exceeded"
                    ),
                    judge=EqualJudge("Complete answer"),
                    prompts=self.prompts()
                )
            outcome = ParsedJournal(path).read()
            assert (outcome["reason"], outcome["calls"], "successful_candidate" in outcome) == (
                "execution_error",
                {
                    "target": 1,
                    "attacker": 0,
                    "judge": 0
                },
                False
            ), "A context error triggered shortening, retry, or a fabricated winner"

    def test_collection_failures_are_errors_even_after_a_negative_assessment(self) -> None:
        for prompts, replies, text in (
            (
                [],
                [],
                None
            ),
            (
                [
                    ProvidedPrompt(
                        "Activate",
                        {
                            "section": "fake",
                            "variant": "Original",
                            "source_index": 41
                        }
                    )
                ],
                [
                    "Activation reply", "No"
                ],
                "No"
            )
        ):
            with self.subTest(text=text), temporary_journal("dan-source-error.jsonl") as path:
                with pytest.raises(
                    ValueError, match="Collection unavailable"
                ):
                    self.attempt(
                        path,
                        ScriptedChat(replies),
                        judge=EqualJudge("Complete answer"),
                        prompts=ScriptedPrompts(
                            prompts, broken=True
                        )
                    )
                outcome = ParsedJournal(path).read()
                assert (outcome["status"], outcome["reason"], outcome["text"], "successful_candidate" in outcome) == (
                    "failure", "execution_error", text, False,
                ), "A collection failure erased the last assessment or became candidate exhaustion"

    def test_assessment_failure_keeps_selection_and_both_exchanges(self) -> None:
        with temporary_journal("dan-judge-error.jsonl") as path:
            with pytest.raises(
                ValueError, match="Malformed judge verdict"
            ):
                self.attempt(
                    path,
                    ScriptedChat(["Activation", "Selected answer"]),
                    judge=QwenJudge(
                        "  Explain café\nexactly.  ",
                        ScriptedChat(["MALFORMED"])
                    ),
                    prompts=self.prompts()
                )
            outcome = ParsedJournal(path).read()
            assert (
                outcome["text"], outcome["status"], outcome["reason"], outcome["assessment"],
                outcome["calls"], "successful_candidate" in outcome,
            ) == (
                "Selected answer", "failure", "assessment_error", None,
                {
                    "target": 2,
                    "attacker": 0,
                    "judge": 1
                },
                False
            ), "Assessment failure erased the selected answer, continued traversal, or claimed a winner"

    def test_new_process_recovers_the_recorded_winner(self) -> None:
        with temporary_journal("dan-process.jsonl") as path:
            self.attempt(
                path,
                ScriptedChat(["Activation", "Complete answer"]),
                judge=EqualJudge("Complete answer"),
                prompts=self.prompts()
            )
            process = subprocess.run(
                [
                    sys.executable, "-c", "import json, sys; from pathlib import Path; from journal.parsed_journal import ParsedJournal; " "print(json.dumps(ParsedJournal(Path(sys.argv[1])).read()))",
                    str(path)
                ],
                check=True, capture_output=True, text=True
            )
            outcome = json.loads(process.stdout)
            assert (outcome["successful_candidate"], outcome["assessment"], outcome["text"]) == (
                {
                    "section": "fake",
                    "variant": "Original",
                    "source_index": 41,
                    "position": 1
                },
                True, "Complete answer"
            ), "A fresh process failed to recover the recorded candidate without reassessing"
