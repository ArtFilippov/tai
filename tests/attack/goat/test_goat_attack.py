"""Bounded pinned GOAT runs with separate controlled role histories."""

import subprocess
import sys
import unittest
from pathlib import Path

import pytest

from attack.goat.goat_attack import GOATAttack
from attack.goat.goat_probe import GOATProbe
from attack.attack_run import AttackRun
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.equal_judge import EqualJudge
from judge.recorded_judge import RecordedJudge
from target.scripted_chat import ScriptedChat
from ...journal_utils import temporary_journal


class TestGOATAttack(unittest.TestCase):
    """Check response-dependent turns and independently assessed selection."""

    def run_attack(self, path: Path, attacker: ScriptedChat, target: ScriptedChat,
                   turns: int) -> str | None:
        journal = VersionedJournal(
            DiskJournal(path)
        )
        return GOATAttack(
            AttackRun(
                PlainBase("Specific request"),
                target,
                RecordedJudge(
                    EqualJudge("Final answer"),
                    journal, "Specific request"
                ),
                journal,
                probe=GOATProbe(
                    "Specific request", attacker, journal, turns
                )
            )
        ).run()

    def test_response_changes_followup(self) -> None:
        attacker = ScriptedChat(["Observation: start\nThought: try\nStrategy: ask\nResponse: First prompt", "safe", "Observation: denied\nThought: adapt\nStrategy: retry\nResponse: Second prompt", "safe"])
        with temporary_journal("goat.jsonl") as path:
            self.run_attack(
                path, attacker,
                ScriptedChat(["Refused", "Final answer"]),
                2
            )
            assert "Previous Model Response (M): Refused" in attacker.transcript(2).last_message("user").text, "The later turn ignored the earlier Target response"

    def test_internal_stop_is_not_final_verdict(self) -> None:
        with temporary_journal("stop.jsonl") as path:
            self.run_attack(
                path, ScriptedChat(["Response: Ask", "unsafe"]),
                ScriptedChat(["Other answer"]), 3
            )
            assert (
                ParsedJournal(path).read()["assessment"],
                ParsedJournal(path).read()["reason"]
            ) == (False, "internal_stop"), "The internal stop replaced the final assessment"

    def test_limit_presents_exact_response(self) -> None:
        with temporary_journal("limit.jsonl") as path:
            self.run_attack(
                path, ScriptedChat(["Response: Ask", "safe"]),
                ScriptedChat(["Final answer"]), 1
            )
            assert (
                ParsedJournal(path).read()["text"],
                ParsedJournal(path).read()["assessment"],
                ParsedJournal(path).read()["reason"]
            ) == ("Final answer", True, "round_limit"), "The bounded run lost its exact presented result"

    def test_missing_first_candidate(self) -> None:
        with temporary_journal("empty.jsonl") as path:
            self.run_attack(
                path,
                ScriptedChat([""]),
                ScriptedChat([]),
                2
            )
            assert ParsedJournal(path).read()["status"] == "no_result", "The absent Target response was presented as an empty result"

    def test_target_history_excludes_attacker_reasoning(self) -> None:
        target = ScriptedChat(["Refused", "Final answer"])
        with temporary_journal("histories.jsonl") as path:
            self.run_attack(
                path,
                ScriptedChat(["Response: First", "safe", "Response: Second", "safe"]),
                target, 2
            )
            assert [(turn.role, turn.content.text) for turn in target.transcript(1).turns] == [
                ("user", "First"), ("assistant", "Refused"), ("user", "Second")
            ], "The Attacker history leaked into the Target conversation"

    def test_execution_error_is_journaled(self) -> None:
        with temporary_journal("error.jsonl") as path:
            with pytest.raises(IndexError):
                self.run_attack(
                    path,
                    ScriptedChat(["Response: Ask"]),
                    ScriptedChat([]),
                    1
                )
            assert ParsedJournal(path).read()["reason"] == "execution_error", "The failed Target exchange was not retained as an error"

    def test_fresh_run_does_not_reuse_first_result(self) -> None:
        with temporary_journal("first.jsonl") as first, temporary_journal("second.jsonl") as second:
            self.run_attack(
                first,
                ScriptedChat(["Response: Ask", "unsafe"]),
                ScriptedChat(["First"]),
                1
            )
            self.run_attack(
                second,
                ScriptedChat(["Response: Ask", "safe"]),
                ScriptedChat(["Second"]),
                1
            )
            assert ParsedJournal(second).read()["text"] == "Second", "The second run reused the earlier selected result"

    def test_cli_module_executes(self) -> None:
        command = subprocess.run(
            [sys.executable, "-m", "main.goat_cli", "--help"],
            capture_output=True, text=True, check=True
        )
        assert "--turns" in command.stdout, "The standalone GOAT module did not expose its bound"

    def test_standalone_cli_runs_offline(self) -> None:
        with temporary_journal("cli.jsonl") as path:
            script = (
                "import sys; import main.goat_cli as cli; "
                "from tests.attack.model_fixture_utils import goat; "
                "cli.snapshot = lambda offline: None; "
                "cli.cuda_device = lambda gpu: 'controlled'; "
                "cli.LazyQwenModel = lambda device, tokens: goat(); "
                "cli.main()"
            )
            subprocess.run(
                [
                    sys.executable, "-c", script,
                    str(path),
                    "Specific request", "--gpu", "0", "--turns", "1"
                ],
                check=True, capture_output=True, text=True
            )
            assert ParsedJournal(path).read()["text"] == "Exact Target answer", "The standalone command did not save the presented Target response"
