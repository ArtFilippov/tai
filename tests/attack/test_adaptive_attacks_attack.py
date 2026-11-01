"""Pinned adaptive resources and independently assessed Target candidates."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest
from garak.probes.adaptive_attacks import ADAPTIVE_TEMPLATE, SUFFIXES_PATH

from attack.adaptive_attacks_probe import AdaptiveAttacksProbe
from attack.assessed_candidates import AssessedCandidates
from attack.attack_run import AttackRun
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.equal_judge import EqualJudge
from judge.recorded_judge import RecordedJudge
from target.scripted_chat import ScriptedChat

BASE = "Explain {a}\nand {b}"


class TestAdaptiveAttacksAttack(unittest.TestCase):
    """Check resource order, result selection, and failure boundaries offline."""

    def run_attack(self, path: Path, replies: list[str], expected: str) -> ScriptedChat:
        """Run one independently journaled attempt with controlled replies.

        Args:
            path (Path): New journal location.
            replies (list[str]): Ordered Target replies.
            expected (str): Text accepted by the judge.

        Returns:
            ScriptedChat: Observed Target conversations.
        """
        journal = VersionedJournal(
            DiskJournal(path)
        )
        target = ScriptedChat(replies)
        _ = AttackRun(
            PlainBase(BASE),
            target,
            RecordedJudge(
                EqualJudge(expected),
                journal, BASE
            ),
            journal,
            probe=AdaptiveAttacksProbe(
                BASE,
                AssessedCandidates(journal)
            )
        ).run()
        return target

    def test_five_native_variants_keep_exact_goal_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.jsonl"
            self.run_attack(
                path, ["no"] * 5, "yes"
            )
            suffixes = [line.strip() for line in SUFFIXES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert [event["prompt"] for event in ParsedJournal(path).events() if event["event"] == "call" and event["role"] == "target"] == [
                ADAPTIVE_TEMPLATE.format(goal=BASE, suffix=suffix) for suffix in suffixes
            ], "The adaptive requests changed the base, template, or suffix order"

    def test_each_candidate_uses_a_fresh_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = self.run_attack(
                Path(directory) / "attempt.jsonl",
                ["no"] * 5, "yes"
            )
            assert [len(history.turns) for history in target.histories] == [1] * 5, "A prior candidate leaked into another Target conversation"

    def test_first_positive_assessment_stops_after_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.jsonl"
            target = self.run_attack(
                path,
                [
                    "no", "yes", "unused"
                ],
                "yes"
            )
            assert (
                len(target.histories),
                sum(1 for event in ParsedJournal(path).events() if event["event"] == "call" and event["role"] == "judge"),
                ParsedJournal(path).read()["text"],
                ParsedJournal(path).read()["assessment"],
                ParsedJournal(path).read()["reason"]
            ) == (
                2, 2, "yes", True, "early_success"
            ), "The first positive verdict did not stop and select its reply"

    def test_judge_uses_the_unmodified_base_for_every_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.jsonl"
            self.run_attack(
                path, ["no"] * 5, "yes"
            )
            assert [event["base"] for event in ParsedJournal(path).events() if event["event"] == "judgment"] == [BASE] * 5, "An adaptive reply was assessed against a transformed request"

    def test_exhaustion_keeps_the_fifth_assessed_reply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.jsonl"
            self.run_attack(
                path,
                [
                    "first", "second", "third", "fourth", "fifth"
                ],
                "never"
            )
            assert (
                ParsedJournal(path).read()["text"],
                ParsedJournal(path).read()["assessment"],
                ParsedJournal(path).read()["reason"]
            ) == (
                "fifth", False, "candidates_exhausted"
            ), "Exhaustion lost the fifth assessed response"

    def test_each_assessment_follows_the_exact_recorded_reply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.jsonl"
            self.run_attack(
                path,
                [
                    "first", "second", "third", "fourth", "fifth"
                ],
                "never"
            )
            assert [event["text"] for event in ParsedJournal(path).events() if event["event"] in {"result", "judgment"}] == [
                text for reply in ("first", "second", "third", "fourth", "fifth") for text in (reply, reply)
            ], "A saved assessment was not linked to its presented full reply"

    def test_failed_exchange_is_not_presented(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.jsonl"
            with pytest.raises(IndexError):
                self.run_attack(
                    path, [], "yes"
                )
            assert ParsedJournal(path).read()["text"] is None, "A failed exchange became a result"

    def test_standalone_module_exposes_limits(self) -> None:
        command = subprocess.run(
            [sys.executable, "-m", "main.adaptive_attacks_cli", "--help"],
            capture_output=True, text=True, check=True
        )
        assert "--tokens" in command.stdout, "The standalone command omitted its token limit"

    def test_standalone_module_runs_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.jsonl"
            script = (
                "import main.adaptive_attacks_cli as cli; "
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
                    BASE, "--gpu", "0"
                ],
                check=True, capture_output=True, text=True
            )
            assert ParsedJournal(path).read()["text"] == "Controlled adaptive reply", "The standalone command lost its final reply"
