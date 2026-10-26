"""Native AutoDAN evolution and terminal selection with controlled roles."""

import unittest
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
import pytest

from attack.autodan_attack import AutoDANAttack
from attack.autodan_budget import AutoDANBudget
from attack.autodan_mutation import AutoDANMutation
from attack.autodan_probe import AutoDANProbe
from attack.autodan_reference import AutoDANReference
from attack.fitd_attack import FITDAttack
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.memory_journal import MemoryJournal
from journal.versioned_journal import VersionedJournal
from judge.equal_judge import EqualJudge
from judge.recorded_judge import RecordedJudge
from main.autodan_cli import launch
from target.scripted_chat import ScriptedChat
from target.scripted_model import ScriptedModel
from ..journal_utils import temporary_journal
from ..target import test_logits


class TestAutoDANAttack(unittest.TestCase):
    """Check genuine native mutation, fitness and no-result outcomes."""

    def attempt(self, path: Path, responses: list[str], attacker: ScriptedChat) -> str | None:
        journal = VersionedJournal(
            DiskJournal(path)
        )
        return FITDAttack(
            PlainBase("Base request"),
            ScriptedChat(responses),
            RecordedJudge(
                EqualJudge("Successful reply"),
                journal, "Base request"
            ),
            journal,
            probe=AutoDANProbe(
                test_logits.TestLogits(),
                journal,
                budget=AutoDANBudget(
                    2, 3, 1, 1, 0.0, 1.0, 7, "Desired completion"
                ),
                mutation=AutoDANMutation(
                    attacker, journal
                ),
                corpus=AutoDANReference()
            )
        ).run()

    def test_native_mutation_changes_next_fitness_population(self) -> None:
        with temporary_journal("ga.jsonl") as path:
            self.attempt(
                path,
                [
                    "I'm sorry", "Successful reply"
                ], ScriptedChat(["Evolved prefix", "Evolved prefix"])
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert any(event.get("action") == "evolve" and "Evolved prefix" in str(event.get("prefixes")) for event in events), "The native GA did not evolve a mutation"

    def test_fitness_changes_selected_prefix_on_second_generation(self) -> None:
        with temporary_journal("evolution.jsonl") as path:
            self.attempt(
                path,
                [
                    "I'm sorry", "Detailed answer"
                ], ScriptedChat(["Evolved prefix", "Evolved prefix"])
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert any(event.get("event") == "call" and event.get("role") == "target" and event.get("prompt") == "Evolved prefix" for event in events), "The evolved prefix never reached the Target"

    def test_result_is_assessed_after_native_prefix_only_success(self) -> None:
        with temporary_journal("result.jsonl") as path:
            self.attempt(
                path,
                [
                    "I'm sorry", "Successful reply"
                ], ScriptedChat(["Evolved prefix", "Evolved prefix"])
            )
            assert (
                ParsedJournal(path).read()["text"],
                ParsedJournal(path).read()["assessment"]
            ) == ("Successful reply", True), "The native successful reply was not linked to its assessment"

    def test_exhaustion_does_not_present_best_candidate(self) -> None:
        with temporary_journal("none.jsonl") as path:
            self.attempt(
                path,
                [
                    "I'm sorry", "I cannot help"
                ], ScriptedChat(["Evolved prefix", "Evolved prefix"])
            )
            assert (
                ParsedJournal(path).read()["status"],
                ParsedJournal(path).read()["text"]
            ) == ("no_result", None), "An unsuccessful candidate was presented"

    def test_failed_mutation_stops_the_native_retry_loop(self) -> None:
        with temporary_journal("mutation-error.jsonl") as path:
            with pytest.raises(RuntimeError):
                self.attempt(
                    path, ["I'm sorry"],
                    ScriptedChat([])
                )
            assert ParsedJournal(path).read()["reason"] == "execution_error", "An unavailable Attacker was retried without a terminal error"

    def test_unsupported_backend_is_rejected_before_inference(self) -> None:
        journal = MemoryJournal()
        attack = AutoDANAttack(
            FITDAttack(
                PlainBase("Base request"),
                ScriptedChat([]),
                EqualJudge("Detailed answer"),
                journal,
                probe=AutoDANProbe(
                    test_logits.TestLogits(),
                    journal,
                    budget=AutoDANBudget(
                        1, 3, 1, 1, 0.0, 0.0, 2, "Target"
                    ),
                    mutation=AutoDANMutation(
                        ScriptedChat([]),
                        journal
                    ),
                    corpus=AutoDANReference()
                )
            ),
            ScriptedModel([]),
            test_logits.TestLogits()
        )
        with pytest.raises(ValueError):
            attack.run()
        assert not journal.events(), "An unsupported backend performed an inference or started a run"

    def test_standalone_command_exposes_all_ga_budgets(self) -> None:
        command = subprocess.run(
            [sys.executable, "-m", "main.autodan_cli", "--help"],
            capture_output=True, text=True, check=True
        )
        assert all(flag in command.stdout for flag in ("--steps", "--batch", "--seed", "--target", "--mutation")), "The standalone command hid a GA budget"

    def test_standalone_command_runs_offline(self) -> None:
        with temporary_journal("cli.jsonl") as path:
            script = (
                "import main.autodan_cli as cli; "
                "from tests.target import test_model, test_logits; "
                "cli.snapshot = lambda offline: None; "
                "cli.cuda_device = lambda gpu: 'controlled'; "
                "cli.LazyQwenModel = lambda device, tokens: test_model.TestModel(); "
                "cli.QwenLogits = lambda assets, device: test_logits.TestLogits(); "
                "cli.main()"
            )
            subprocess.run(
                [
                    sys.executable, "-c", script,
                    str(path),
                    "Base request", "--gpu", "0", "--steps", "1", "--target", "Desired completion"
                ],
                check=True, capture_output=True, text=True
            )
            assert ParsedJournal(path).read()["text"] == "Detailed answer", "The offline standalone command lost its selected result"

    def test_invalid_budget_precedes_model_access(self) -> None:
        with temporary_journal("invalid.jsonl") as path:
            arguments = Namespace(
                journal=path, base="Base request", gpu=0, tokens=32,
                steps=1, batch=3, elites=3, points=1,
                crossover=0.5, mutation=0.1, seed=7, target="Desired"
            )
            with pytest.raises(ValueError):
                launch(arguments)
            assert not path.exists(), "An invalid GA budget reserved a journal before rejection"
