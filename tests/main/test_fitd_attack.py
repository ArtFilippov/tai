"""End-to-end controlled runs of the pinned FITD strategy."""

import json
from pathlib import Path

from fitd.disk_journal import DiskJournal
from fitd.equal_judge import EqualJudge
from fitd.parsed_journal import ParsedJournal
from fitd.run_outcome import RunOutcome
from fitd.scripted_chat import ScriptedChat
from main.fitd_attack import FITDAttack


class TestFITDAttack:
    """Exercises real garak escalation against separate scripted roles."""

    def run(self, path: Path, base: str, response: str) -> RunOutcome:
        attacker = ScriptedChat(["Safe opening", "First escalation", "Second escalation", "Third escalation", "yes", "yes", "yes", "yes", "unsafe"])
        target = ScriptedChat(["First", "Second", "Third", "Fourth", response])
        FITDAttack(
            target, attacker,
            EqualJudge(response),
            DiskJournal(path),
            turns=8
        ).run(base)
        return ParsedJournal(path).read()

    def test_presented_result(self, tmp_path: Path) -> None:
        outcome = self.run(
            tmp_path / "first.jsonl", "Request alpha", "Exact alpha"
        )
        assert outcome["text"] == "Exact alpha", "The attack did not present the last accepted response"

    def test_judgment(self, tmp_path: Path) -> None:
        outcome = self.run(
            tmp_path / "first.jsonl", "Request alpha", "Exact alpha"
        )
        assert outcome["assessment"] is True, "The exact presented result was not judged"

    def test_escales_to_base(self, tmp_path: Path) -> None:
        path = tmp_path / "steps.jsonl"
        self.run(
            path, "Request beta", "Exact beta"
        )
        steps = [json.loads(line)["prompt"] for line in path.read_text(encoding="utf-8").splitlines()
                 if json.loads(line)["event"] == "step"]
        assert steps == ["Safe opening", "Second escalation", "Third escalation", "Request beta", "Request beta"], "The pinned strategy did not escalate to the base request"

    def test_independent_sequential_runs(self, tmp_path: Path) -> None:
        self.run(
            tmp_path / "first.jsonl", "Request alpha", "Exact alpha"
        )
        outcome = self.run(
            tmp_path / "second.jsonl", "Request beta", "Exact beta"
        )
        assert outcome["text"] == "Exact beta", "The second base request reused the first result"

    def test_call_counts(self, tmp_path: Path) -> None:
        outcome = self.run(
            tmp_path / "counts.jsonl", "Request gamma", "Exact gamma"
        )
        assert outcome["calls"] == {
            "target": 5,
            "attacker": 9,
            "judge": 1
        }, "The roles' calls were not recovered"

    def test_roles_have_separate_histories(self, tmp_path: Path) -> None:
        path = tmp_path / "roles.jsonl"
        self.run(
            path, "Request delta", "Exact delta"
        )
        steps = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if json.loads(line)["event"] == "step"]
        assert [turn["text"] for turn in steps[-1]["turns"] if turn["role"] == "user"] == ["Safe opening", "Second escalation", "Third escalation", "Request delta", "Request delta"], "Attacker evaluation leaked into the Target conversation"

    def test_empty_target_result(self, tmp_path: Path) -> None:
        outcome = self.run(
            tmp_path / "empty.jsonl", "Request empty", ""
        )
        assert outcome["text"] is not None and not outcome["text"], "An empty completed response was treated as absent"

    def test_no_initial_turn(self, tmp_path: Path) -> None:
        path = tmp_path / "no-result.jsonl"
        FITDAttack(
            ScriptedChat([]),
            ScriptedChat(["I cannot fulfill your request"] * 3),
            EqualJudge("Nothing"),
            DiskJournal(path),
            turns=8
        ).run("Request refused")
        assert ParsedJournal(path).read()["status"] == "no_result", "A failed opening was presented as a result"

    def test_refusal_has_no_presented_result(self, tmp_path: Path) -> None:
        path = tmp_path / "refusal.jsonl"
        FITDAttack(
            ScriptedChat(["I cannot fulfill your request"]),
            ScriptedChat(["Safe opening", "First", "Second", "Third", "Safer opening"]),
            EqualJudge("Nothing"),
            DiskJournal(path),
            turns=1
        ).run("Request refused")
        assert ParsedJournal(path).read()["calls"]["judge"] == 0, "The judge received a refused Target response"
