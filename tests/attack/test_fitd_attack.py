"""End-to-end controlled runs of the pinned FITD strategy."""

import json
import subprocess
import sys
import unittest
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from attack.fitd_attack import FITDAttack
from attack.fitd_probe import FITDProbe
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from judge.equal_judge import EqualJudge
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.continuing_chat import ContinuingChat
from target.scripted_chat import ScriptedChat
from ..journal_utils import temporary_journal


class TestFITDAttack(unittest.TestCase):
    """Exercises real garak escalation against separate scripted roles."""

    def file(self, name: str) -> AbstractContextManager[Path]:
        return temporary_journal(name)

    def attack(self, path: Path, base: str, response: str) -> Mapping[str, object]:
        attacker = ScriptedChat(["Safe opening", "First escalation", "Second escalation", "Third escalation", "yes", "yes", "yes", "yes", "unsafe"])
        target = ScriptedChat(["First", "Second", "Third", "Fourth", response])
        journal = DiskJournal(path)
        FITDAttack(
            target,
            RecordedJudge(
                EqualJudge(response), journal, base
            ),
            journal,
            FITDProbe(
                base, ContinuingChat(attacker), journal, 8
            )
        ).run(base)
        return ParsedJournal(path).read()

    def failed(self, path: Path, replies: list[str]) -> Mapping[str, object]:
        journal = DiskJournal(path)
        with pytest.raises(
            (
                ValueError, IndexError
            )
        ):
            FITDAttack(
                ScriptedChat(["Exact candidate"]),
                RecordedJudge(
                    QwenJudge(
                        "Request",
                        ScriptedChat(replies)
                    ),
                    journal, "Request"
                ),
                journal,
                FITDProbe(
                    "Request",
                    ContinuingChat(
                        ScriptedChat(["Opening", "One", "Two", "Three"])
                    ),
                    journal, 1
                )
            ).run("Request")
        return ParsedJournal(path).read()

    def mismatched(self, path: Path) -> Mapping[str, object]:
        journal = DiskJournal(path)
        with pytest.raises(
            ValueError, match="base request differs"
        ):
            FITDAttack(
                ScriptedChat([]),
                RecordedJudge(
                    EqualJudge("Nothing"),
                    journal, "Other request"
                ),
                journal,
                FITDProbe(
                    "First request",
                    ContinuingChat(
                        ScriptedChat([])
                    ),
                    journal, 1
                )
            ).run("Other request")
        return ParsedJournal(path).read()

    def test_presented_result(self) -> None:
        with self.file("first.jsonl") as path:
            outcome = self.attack(
                path, "Request alpha", "Exact alpha"
            )
            assert outcome["text"] == "Exact alpha", "The attack did not present the last accepted response"

    def test_judgment(self) -> None:
        with self.file("first.jsonl") as path:
            outcome = self.attack(
                path, "Request alpha", "Exact alpha"
            )
            assert outcome["assessment"] is True, "The exact presented result was not judged"

    def test_round_limit(self) -> None:
        with self.file("limit.jsonl") as path:
            journal = DiskJournal(path)
            FITDAttack(
                ScriptedChat(["Partial"]),
                RecordedJudge(
                    EqualJudge("Success"),
                    journal, "Request"
                ),
                journal,
                FITDProbe(
                    "Request",
                    ContinuingChat(
                        ScriptedChat(["Safe", "One", "Two", "Three", "yes", "unsafe"])
                    ),
                    journal, 1
                )
            ).run("Request")
            assert ParsedJournal(path).read()["status"] == "round_limit", "The round limit was mistaken for success"

    def test_escales_to_base(self) -> None:
        with self.file("steps.jsonl") as path:
            self.attack(
                path, "Request beta", "Exact beta"
            )
            steps = [json.loads(line)["prompt"] for line in path.read_text(encoding="utf-8").splitlines()
                     if json.loads(line)["event"] == "step"]
            assert steps == ["Safe opening", "Second escalation", "Third escalation", "Request beta", "Request beta"], "The pinned strategy did not escalate to the base request"

    def test_independent_sequential_runs(self) -> None:
        with self.file("first.jsonl") as first, self.file("second.jsonl") as second:
            self.attack(
                first, "Request alpha", "Exact alpha"
            )
            outcome = self.attack(
                second, "Request beta", "Exact beta"
            )
            assert outcome["text"] == "Exact beta", "The second base request reused the first result"

    def test_call_counts(self) -> None:
        with self.file("counts.jsonl") as path:
            outcome = self.attack(
                path, "Request gamma", "Exact gamma"
            )
            assert outcome["calls"] == {
                "target": 5,
                "attacker": 8,
                "judge": 5
            }, "The roles' calls were not recovered"

    def test_roles_have_separate_histories(self) -> None:
        with self.file("roles.jsonl") as path:
            self.attack(
                path, "Request delta", "Exact delta"
            )
            steps = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                     if json.loads(line)["event"] == "step"]
            assert [turn["text"] for turn in steps[-1]["turns"] if turn["role"] == "user"] == ["Safe opening", "Second escalation", "Third escalation", "Request delta", "Request delta"], "Attacker evaluation leaked into the Target conversation"

    def test_empty_target_result(self) -> None:
        with self.file("empty.jsonl") as path:
            outcome = self.attack(
                path, "Request empty", ""
            )
            assert outcome["text"] is not None and not outcome["text"], "An empty completed response was treated as absent"

    def test_no_initial_turn(self) -> None:
        with self.file("no-result.jsonl") as path:
            journal = DiskJournal(path)
            FITDAttack(
                ScriptedChat([]),
                RecordedJudge(
                    EqualJudge("Nothing"),
                    journal, "Request refused"
                ),
                journal,
                FITDProbe(
                    "Request refused",
                    ContinuingChat(
                        ScriptedChat(["I cannot fulfill your request"] * 3)
                    ),
                    journal, 8
                )
            ).run("Request refused")
            assert ParsedJournal(path).read()["status"] == "no_result", "A failed opening was presented as a result"

    def test_refusal_has_no_presented_result(self) -> None:
        with self.file("refusal.jsonl") as path:
            journal = DiskJournal(path)
            FITDAttack(
                ScriptedChat(["I cannot fulfill your request"]),
                RecordedJudge(
                    EqualJudge("Nothing"),
                    journal, "Request refused"
                ),
                journal,
                FITDProbe(
                    "Request refused",
                    ContinuingChat(
                        ScriptedChat(["Safe opening", "First", "Second", "Third", "Safer opening"])
                    ),
                    journal, 1
                )
            ).run("Request refused")
            assert ParsedJournal(path).read()["calls"] == {
                "target": 1,
                "attacker": 5,
                "judge": 0
            }, "The judge received a refused Target response"

    def test_command(self) -> None:
        with self.file("command.jsonl") as path:
            subprocess.run(
                [
                    str(
                        Path(sys.executable).with_name("tai")
                    ),
                    str(path),
                    "Explain the topic in detail"
                ],
                check=True, capture_output=True, text=True
            )
            assert ParsedJournal(path).read()["status"] == "success", "The command did not finish the experiment"

    def test_inconsistent_base(self) -> None:
        with self.file("mismatch.jsonl") as path:
            assert self.mismatched(path)["reason"] == "execution_error", "The probe failure was misreported as an assessment error"

    def test_safer_opening_discards_refusal(self) -> None:
        with self.file("safer.jsonl") as path:
            journal = DiskJournal(path)
            FITDAttack(
                ScriptedChat(["I cannot fulfill your request", "Fresh reply"]),
                RecordedJudge(
                    EqualJudge("Never"),
                    journal, "Request"
                ),
                journal,
                FITDProbe(
                    "Request",
                    ContinuingChat(
                        ScriptedChat(["Opening", "One", "Two", "Three", "Safer opening", "yes"])
                    ),
                    journal, 2
                )
            ).run("Request")
            events = list(
                ParsedJournal(path).events()
            )
            steps = [event for event in events if event.get("event") == "step"]
            assert [(event["discarded"], event["next"], steps[1]["prompt"])
                    for event in events if event.get("action") == "safer"] == [
                        (
                            [
                                {
                                    "role": "user",
                                    "text": "Opening"
                                },
                                {
                                    "role": "assistant",
                                    "text": "I cannot fulfill your request"
                                }
                            ],
                            "Safer opening", "Safer opening"
                        )
                    ], "The refused opening was not discarded before the next request"

    def test_paraphrase_restores_accepted_history(self) -> None:
        with self.file("paraphrase.jsonl") as path:
            journal = DiskJournal(path)
            FITDAttack(
                ScriptedChat(["First", "I cannot fulfill your request", "Recovered"]),
                RecordedJudge(
                    EqualJudge("Never"),
                    journal, "Request"
                ),
                journal,
                FITDProbe(
                    "Request",
                    ContinuingChat(
                        ScriptedChat(["Opening", "One", "Two", "Three", "yes", "Paraphrased", "yes"])
                    ),
                    journal, 3
                )
            ).run("Request")
            events = list(
                ParsedJournal(path).events()
            )
            steps = [event for event in events if event.get("event") == "step"]
            assert steps[-1]["turns"] == [
                {
                    "role": "user",
                    "text": "Opening"
                },
                {
                    "role": "assistant",
                    "text": "First"
                },
                {
                    "role": "user",
                    "text": "Paraphrased"
                },
                {
                    "role": "assistant",
                    "text": "Recovered"
                }
            ], "The paraphrase did not restore the earlier accepted conversation"

    def test_discarded_reply_does_not_replace_result(self) -> None:
        with self.file("discarded.jsonl") as path:
            journal = DiskJournal(path)
            FITDAttack(
                ScriptedChat(["First", "I cannot fulfill your request"]),
                RecordedJudge(
                    EqualJudge("Never"),
                    journal, "Request"
                ),
                journal,
                FITDProbe(
                    "Request",
                    ContinuingChat(
                        ScriptedChat(["Opening", "One", "Two", "Three", "yes", "Paraphrased"])
                    ),
                    journal, 2
                )
            ).run("Request")
            assert ParsedJournal(path).read()["text"] == "First", "The discarded refusal replaced the selected result"

    def test_realignment_keeps_conversation(self) -> None:
        with self.file("realign.jsonl") as path:
            journal = DiskJournal(path)
            FITDAttack(
                ScriptedChat(["Off topic", "Improved"]),
                RecordedJudge(
                    EqualJudge("Improved"),
                    journal, "Request"
                ),
                journal,
                FITDProbe(
                    "Request",
                    ContinuingChat(
                        ScriptedChat(["Opening", "One", "Two", "Three", "no"])
                    ),
                    journal, 2
                )
            ).run("Request")
            events = list(
                ParsedJournal(path).events()
            )
            steps = [event for event in events if event.get("event") == "step"]
            assert [(event["action"], event["discarded"], event["next"] == steps[1]["prompt"])
                    for event in events if event.get("event") == "decision"] == [
                        ("realignment", [], True)
                    ], "Realignment discarded the active conversation"

    def test_invalid_verdict_is_saved_as_failure(self) -> None:
        with self.file("invalid.jsonl") as path:
            assert self.failed(path, ["MAYBE"])["assessment_error"] == "ValueError(\"Malformed judge verdict: 'MAYBE'\")", "The malformed verdict was recorded as a negative assessment"

    def test_judge_exception_preserves_candidate(self) -> None:
        with self.file("exception.jsonl") as path:
            assert self.failed(path, [])["text"] == "Exact candidate", "The judge failure lost the submitted result"

    def test_sequential_metrics_are_independent(self) -> None:
        with self.file("first.jsonl") as first, self.file("second.jsonl") as second:
            self.attack(
                first, "Request alpha", "Exact alpha"
            )
            self.attack(
                second, "Request beta", "Exact beta"
            )
            assert ParsedJournal(second).read()["calls"] == {
                "target": 5,
                "attacker": 8,
                "judge": 5
            }, "The second run included calls from the first"
