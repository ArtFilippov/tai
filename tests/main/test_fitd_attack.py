"""End-to-end controlled runs of the pinned FITD strategy."""

import json
import unittest
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path

from fitd.disk_journal import DiskJournal
from fitd.equal_judge import EqualJudge
from fitd.parsed_journal import ParsedJournal
from fitd.scripted_chat import ScriptedChat
from main.fitd_attack import FITDAttack
from ..journal_utils import temporary_journal


class TestFITDAttack(unittest.TestCase):
    """Exercises real garak escalation against separate scripted roles."""

    def file(self, name: str) -> AbstractContextManager[Path]:
        return temporary_journal(name)

    def attack(self, path: Path, base: str, response: str) -> Mapping[str, object]:
        attacker = ScriptedChat(["Safe opening", "First escalation", "Second escalation", "Third escalation", "yes", "yes", "yes", "yes", "unsafe"])
        target = ScriptedChat(["First", "Second", "Third", "Fourth", response])
        FITDAttack(
            target, attacker,
            EqualJudge(response),
            DiskJournal(path),
            turns=8
        ).run(base)
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
                "attacker": 9,
                "judge": 1
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
            FITDAttack(
                ScriptedChat([]),
                ScriptedChat(["I cannot fulfill your request"] * 3),
                EqualJudge("Nothing"),
                DiskJournal(path),
                turns=8
            ).run("Request refused")
            assert ParsedJournal(path).read()["status"] == "no_result", "A failed opening was presented as a result"

    def test_refusal_has_no_presented_result(self) -> None:
        with self.file("refusal.jsonl") as path:
            FITDAttack(
                ScriptedChat(["I cannot fulfill your request"]),
                ScriptedChat(["Safe opening", "First", "Second", "Third", "Safer opening"]),
                EqualJudge("Nothing"),
                DiskJournal(path),
                turns=1
            ).run("Request refused")
            assert ParsedJournal(path).read()["calls"] == {
                "target": 1,
                "attacker": 5,
                "judge": 0
            }, "The judge received a refused Target response"
