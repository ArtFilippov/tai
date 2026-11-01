"""Native AutoDAN evolution and terminal selection with controlled roles."""

import unittest
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from attack.autodan_attack import AutoDANAttack
from attack.autodan_budget import AutoDANBudget
from attack.autodan_corpus import AutoDANCorpus
from attack.autodan_mutation import AutoDANMutation
from attack.autodan_probe import AutoDANProbe
from attack.autodan_reference import AutoDANReference
from attack.attack_run import AttackRun
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.memory_journal import MemoryJournal
from journal.versioned_journal import VersionedJournal
from judge.equal_judge import EqualJudge
from judge.recorded_judge import RecordedJudge
from main.autodan_cli import bounds, launch
from target.logits import Logits
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
        return AttackRun(
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
                    2, 3, 1, 1, 0.0, 1.0, 7, "Desired completion", "ga"
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
            AttackRun(
                PlainBase("Base request"),
                ScriptedChat([]),
                EqualJudge("Detailed answer"),
                journal,
                probe=AutoDANProbe(
                    test_logits.TestLogits(),
                    journal,
                    budget=AutoDANBudget(
                        1, 3, 1, 1, 0.0, 0.0, 2, "Target", "hga"
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
        assert all(flag in command.stdout for flag in ("--mode", "--steps", "--batch", "--seed", "--target", "--mutation")), "The standalone command hid an AutoDAN budget"

    def hga(self, path: Path, responses: list[str], rate: float) -> str | None:
        """Run the real native hierarchy with deterministic local word choices.

        Args:
            path (Path): Independent journal destination.
            responses (list[str]): Target script.
            rate (float): Native Attacker mutation probability.

        Returns:
            str | None: Presented Target reply, if successful.
        """
        journal = VersionedJournal(
            DiskJournal(path)
        )
        corpus = Mock(spec=AutoDANCorpus)
        corpus.population.return_value = ["bright", "shiny", "dull"]
        corpus.description.return_value = "controlled"
        logits = Mock(spec=Logits)
        logits.score.side_effect = [0.0, 1.0, 2.0, 2.0, 0.0, 0.0]
        logits.profile.return_value = {
            "id": "Qwen/Qwen2.5-0.5B-Instruct"
        }
        with patch(
            "garak.resources.autodan.genetic.get_synonyms", return_value=["shiny"]
        ), patch(
            "garak.resources.autodan.genetic.nltk.word_tokenize", side_effect=str.split
        ):
            return AttackRun(
                PlainBase("Base request"),
                ScriptedChat(responses),
                RecordedJudge(
                    EqualJudge("Successful reply"), journal, "Base request"
                ),
                journal,
                probe=AutoDANProbe(
                    logits, journal,
                    budget=AutoDANBudget(
                        2, 3, 1, 1, 1.0, rate, 7, "Desired", "hga"
                    ),
                    mutation=AutoDANMutation(
                        ScriptedChat(["Evolved prefix", "Evolved prefix"]), journal
                    ),
                    corpus=corpus
                )
            ).run()

    def test_hga_word_replacement_reaches_next_target_call(self) -> None:
        with temporary_journal("hga-transition.jsonl") as path:
            self.hga(
                path,
                [
                    "I'm sorry", "Successful reply"
                ], rate=0.0
            )
            events = list(
                ParsedJournal(path).events()
            )
            momentum = {
                "bright": 0.0,
                "shiny": -1.0,
                "dull": -2.0
            }
            evolution = [event for event in events if event.get("action") == "evolve"]
            expected = ["bright", "shiny", "shiny"]
            assert evolution[0].get("momentum") == momentum and evolution[0].get("prefixes") == expected and [event["prompt"] for event in events if event.get("role") == "target"] == ["bright", "shiny"], "The native HGA transition did not reach the Target"

    def test_hga_success_is_assessed_without_base_in_target_prompt(self) -> None:
        with temporary_journal("hga-success.jsonl") as path:
            self.hga(
                path,
                [
                    "I'm sorry", "Successful reply"
                ], rate=0.0
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert (
                ParsedJournal(path).read()["assessment"],
                [event["prompt"] for event in events if event.get("role") == "target"]
            ) == (True, ["bright", "shiny"]), "The HGA result or prefix-only success check was lost"

    def test_hga_native_mutation_uses_independent_attacker(self) -> None:
        with temporary_journal("hga-mutation.jsonl") as path:
            self.hga(
                path, ["I'm sorry", "Successful reply"], rate=1.0
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert [event["role"] for event in events if event.get("event") == "call"] == ["target", "attacker", "attacker", "target", "judge"], "The native HGA did not use a separate mutation Attacker"

    def test_hga_exhaustion_retains_search_without_result(self) -> None:
        with temporary_journal("hga-none.jsonl") as path:
            self.hga(
                path,
                [
                    "I'm sorry", "I cannot help"
                ], rate=0.0
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert (
                ParsedJournal(path).read()["status"],
                ParsedJournal(path).read()["text"],
                len([event for event in events if event.get("action") == "fitness"]),
                "'mode': 'hga'" in str(events[0]["attack"]),
                any(event.get("event") == "result" for event in events)
            ) == ("no_result", None, 2, True, False), "The exhausted HGA lost search activity or invented a result"

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
                    "Base request", "--gpu", "0", "--steps", "1", "--mode", "hga", "--target", "Desired completion"
                ],
                check=True, capture_output=True, text=True
            )
            assert (
                ParsedJournal(path).read()["text"],
                "'type': 'autodan_hga'" in str(list(ParsedJournal(path).events())[0]["attack"])
            ) == ("Detailed answer", True), "The offline standalone command lost its HGA selection"

    def test_invalid_budget_precedes_model_access(self) -> None:
        with temporary_journal("invalid.jsonl") as path:
            arguments = Namespace(
                journal=path, base="Base request", gpu=0, tokens=32,
                steps=1, batch=3, elites=3, points=1,
                crossover=0.5, mutation=0.1, seed=7, target="Desired", mode="hga"
            )
            with pytest.raises(ValueError):
                launch(arguments)
            assert not path.exists(), "An invalid GA budget reserved a journal before rejection"

    def test_hga_allows_odd_offspring_without_ga_crossover(self) -> None:
        arguments = Namespace(
            steps=2, batch=4, elites=1, points=1, crossover=0.5,
            mutation=0.0, seed=7, target="Desired", mode="hga"
        )
        assert bounds(arguments).batch == 4, "HGA inherited the GA-only even offspring constraint"
