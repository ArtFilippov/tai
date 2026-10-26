"""Native deletion order, journal linkage, bounds and standalone command."""

import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
from attack.deletion_native import NativeDeletion

from attack.assessed_candidates import AssessedCandidates
from attack.deletion_budget import DeletionBudget
from attack.deletion_limits import DeletionLimits
from attack.deletion_probe import DeletionProbe
from attack.fitd_attack import FITDAttack
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.equal_judge import EqualJudge
from judge.recorded_judge import RecordedJudge
from main.deletion_cli import launch
from target.scripted_chat import ScriptedChat


class TestDeletionAttack(unittest.TestCase):
    """Exercise native generation through complete journaled attempts."""

    def attempt(self, path: Path, base: str, budget: DeletionLimits, replies: list[str], expected: str = "yes") -> ScriptedChat:
        """Run one attempt using controlled Target replies.

        Args:
            path (Path): Fresh journal destination.
            base (str): Original request.
            budget (DeletionLimits): Native bounds.
            replies (list[str]): Target completions.
            expected (str, default="yes"): Positive completion.

        Returns:
            ScriptedChat: Observed Target conversations.
        """
        journal = VersionedJournal(
            DiskJournal(path)
        )
        target = ScriptedChat(replies)
        _ = FITDAttack(
            PlainBase(base),
            target,
            RecordedJudge(
                EqualJudge(expected),
                journal, base
            ),
            journal,
            probe=DeletionProbe(
                base,
                AssessedCandidates(journal),
                journal, budget=budget,
                variants=NativeDeletion(
                    budget.perturbations, budget.positions, budget.ascii
                )
            )
        ).run()
        return target

    def test_native_order_metadata_and_unicode_backspaces(self) -> None:
        base = "é猫"
        budget: DeletionLimits = DeletionBudget(
            2, 3, 2
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            target = self.attempt(
                path, base, budget, ["no"] * 18
            )
            expected = list(
                NativeDeletion(
                    budget.perturbations, budget.positions, budget.ascii
                ).generate(base)
            )
            events = list(
                ParsedJournal(path).events()
            )
            decisions = [event for event in events if event["event"] == "decision"]
            calls = [event for event in events if event["event"] == "call" and event["role"] == "target"]
            metadata = [
                {
                    "bad_character_category": "deletion",
                    "perturbation_count": variant["count"],
                    "operations": variant["operations"]
                }
                for variant in expected
            ]
            assert (
                [(event["prompt"], event["perturbation_count"], event["operations"]) for event in decisions],
                [event["prompt"] for event in calls],
                [len(history.turns) for history in target.histories],
                [event["base"] for event in events if event["event"] == "judgment"],
                [history.turns[0].content.notes for history in target.histories],
                [event["metadata"] for event in calls]
            ) == (
                [(variant["text"], variant["count"], variant["operations"]) for variant in expected], [variant["text"] for variant in expected],
                [1] * len(expected),
                [base] * len(expected),
                metadata, metadata
            ), "Native deletion order, metadata, Unicode, or isolated Target transmissions diverged"

    def test_native_position_and_ascii_order_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.attempt(
                path, "é猫",
                DeletionBudget(
                    1, 3, 2
                ),
                ["no"] * 6
            )
            events = [event for event in ParsedJournal(path).events() if event["event"] == "decision"]
            assert [(event["prompt"], event["operations"]) for event in events] == [
                (
                    " \bé猫",
                    {
                        "positions": [0],
                        "ascii_codes": [32]
                    }
                ),
                (
                    "~\bé猫",
                    {
                        "positions": [0],
                        "ascii_codes": [126]
                    }
                ),
                (
                    "é \b猫",
                    {
                        "positions": [1],
                        "ascii_codes": [32]
                    }
                ),
                (
                    "é~\b猫",
                    {
                        "positions": [1],
                        "ascii_codes": [126]
                    }
                ),
                (
                    "é猫 \b",
                    {
                        "positions": [2],
                        "ascii_codes": [32]
                    }
                ),
                (
                    "é猫~\b",
                    {
                        "positions": [2],
                        "ascii_codes": [126]
                    }
                )
            ], "Native deletion position or ASCII order changed"

    def test_short_input_uses_native_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.attempt(
                path, "",
                DeletionBudget(
                    1, 8, 4
                ),
                ["no"] * 4
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert [event["prompt"] for event in events if event["event"] == "decision"] == [f"{char}\b" for char in (" ", "?", "^", "}")], "The short input did not preserve native ASCII sampling"

    def test_default_limits_bound_a_long_request_to_32_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            target = self.attempt(
                path, "long enough",
                DeletionBudget(
                    1, 8, 4
                ),
                ["no"] * 32
            )
            assert len(target.histories) == 32, "Default deletion bounds did not produce 32 independent Target requests"

    def test_first_positive_presents_exact_reply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            target = self.attempt(
                path, "abc",
                DeletionBudget(
                    1, 3, 2
                ),
                [
                    "no", "yes", "unused"
                ]
            )
            result = ParsedJournal(path).read()
            assert (
                len(target.histories),
                result["text"], result["assessment"], result["reason"]
            ) == (2, "yes", True, "early_success"), "The first positive deletion reply was not presented"

    def test_exhaustion_presents_last_assessed_reply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.attempt(
                path, "a",
                DeletionBudget(
                    1, 8, 2
                ),
                [
                    "one", "two", "three", "last"
                ]
            )
            result = ParsedJournal(path).read()
            assert (
                result["text"], result["assessment"], result["reason"],
                ParsedJournal(path).calls(
                    ParsedJournal(path).events()
                )["target"]
            ) == ("last", False, "candidates_exhausted", 4), "Exhaustion did not retain the last assessed reply"

    def test_invalid_limits_fail_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            with pytest.raises(ValueError):
                self.attempt(
                    path, "base",
                    DeletionBudget(
                        True, 8, 4
                    ),
                    ["unused"]
                )
            assert not ParsedJournal(path).calls(
                ParsedJournal(path).events()
            )["target"], "A boolean deletion limit reached Target inference"

    def test_unsupported_limits(self) -> None:
        budgets = [
            DeletionBudget(
                0, 8, 4
            ),
            DeletionBudget(
                1, 0, 4
            ),
            DeletionBudget(
                1, 8, 96
            ),
            DeletionBudget(
                2, 1, 4
            ),
            DeletionBudget(
                2, 24, 95
            ),
            DeletionBudget(
                *json.loads("[1.0, 8, 4]")
            )
        ]
        assert all(self.invalid(budget) for budget in budgets), "An unsupported deletion bound was accepted"

    def invalid(self, budget: DeletionBudget) -> bool:
        """Check validation rejection without reaching model inference.

        Args:
            budget (DeletionBudget): Invalid requested limits.

        Returns:
            bool: Whether validation rejects them.
        """
        try:
            budget.validate()
        except ValueError:
            return True
        return False

    def test_target_error_has_no_presented_reply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            with pytest.raises(IndexError):
                self.attempt(
                    path, "abc",
                    DeletionBudget(
                        1, 1, 1
                    ),
                    []
                )
            assert (
                ParsedJournal(path).read()["status"],
                ParsedJournal(path).read()["text"]
            ) == ("failure", None), "A failed deletion call became a result"

    def test_cli_rejects_before_reserving_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "main.deletion_cli.snapshot",
            side_effect=AssertionError("Model accessed")
        ):
            path = Path(directory) / "run.jsonl"
            with pytest.raises(ValueError):
                launch(
                    Namespace(
                        journal=path, base="abc", gpu=0, tokens=192, perturbations=1, positions=8, ascii=False
                    )
                )
            assert not path.exists(), "Invalid CLI limits reserved a journal"

    def test_cli_restart_preserves_existing_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "main.deletion_cli.snapshot", side_effect=AssertionError("Model accessed")
        ):
            path = Path(directory) / "run.jsonl"
            path.write_text(
                "saved result", encoding="utf-8"
            )
            with pytest.raises(FileExistsError):
                launch(
                    Namespace(
                        journal=path, base="abc", gpu=0, tokens=192,
                        perturbations=1, positions=8, ascii=4
                    )
                )
            assert path.read_text(encoding="utf-8") == "saved result", "Restart replaced the existing deletion journal"

    def test_standalone_command_runs_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            script = (
                "import main.deletion_cli as cli; "
                "from tests.attack.model_fixture_utils import adaptive; "
                "cli.snapshot = lambda offline: None; "
                "cli.cuda_device = lambda gpu: 'controlled'; "
                "cli.LazyQwenModel = lambda device, tokens: adaptive(); "
                "cli.main()"
            )
            subprocess.run(
                [
                    sys.executable, "-c", script,
                    str(path),
                    "abc", "--gpu", "0", "--positions", "2", "--ascii", "2"
                ],
                check=True, capture_output=True, text=True
            )
            assert (
                ParsedJournal(path).calls(ParsedJournal(path).events())["target"],
                '"budget": {"perturbations": 1, "positions": 2, "ascii": 2}' in json.dumps(next(ParsedJournal(path).events())["attack"])
            ) == (
                4,
                True
            ), "The deletion command lost its effective bounds"
