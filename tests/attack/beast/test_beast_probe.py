"""BEAST presents and judges only the final Target response."""

import subprocess
import sys
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import torch

from attack.attack_run import AttackRun
from attack.beast.configured_beast_budget import ConfiguredBeastBudget
from attack.beast.beast_probe import BeastProbe
from attack.beast.beam_search import BeamSearch
from attack.beast.envelopes.beast_attack_envelope import BeastAttackEnvelope
from attack.desired_response.provided_desired_response import ProvidedDesiredResponse
from attack.shared_model_attack_setup import SharedModelAttackSetup
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from journal.memory_journal import MemoryJournal
from judge.judge import Judge
from judge.recorded_judge import RecordedJudge
from target.scripted_chat import ScriptedChat
from target.model import Model
from target.token_probabilities import TokenProbabilities
from ...journal_utils import temporary_journal


class TestBeastProbe(unittest.TestCase):
    """Link selected suffix evidence separately from the judged reply."""

    def test_non_qwen_target_runs_with_the_token_contract(self) -> None:
        target = Mock(spec=Model)
        target.settings.return_value = {
            "id": "another causal model",
            "revision": "different"
        }
        def respond(messages: Sequence[dict[str, str]]) -> str:
            return "NO" if messages[0]["role"] == "system" else "A reply"
        target.generate.side_effect = respond
        tokens = Mock(spec=TokenProbabilities)
        tokens.profile.return_value = {"id": "another causal model"}
        tokens.decode.return_value = " suffix"
        distribution = torch.tensor([0.0, 1.0])
        tokens.distribution.return_value = [distribution]
        tokens.loss.return_value = [0.2]
        base = PlainBase("Explain café")
        journal = MemoryJournal()
        result = BeastAttackEnvelope(
            base,
            SharedModelAttackSetup(
                target,
                base.text()
            ),
            journal,
            BeastProbe(
                base.text(),
                ProvidedDesiredResponse(
                    base, "A café serves coffee"
                ),
                BeamSearch(
                    tokens,
                    ConfiguredBeastBudget(
                        1, 1, 1, 1
                    )
                ),
                journal
            )
        ).run()
        assert result == "A reply", "BEAST rejected a model that satisfied its contracts"

    def attempt(self, path: Path, tokens: Mock, target: ScriptedChat,
                judge: Judge, distribution: torch.Tensor, loss: float) -> str | None:
        """Run a controlled search against one exact request.

        Args:
            path (Path): Independent journal path.
            tokens (Mock): Scripted token capability.
            target (ScriptedChat): Final Target reply.
            judge (Judge): Independent assessment.
            distribution (torch.Tensor): One controlled suffix distribution.
            loss (float): One controlled desired-answer loss.

        Returns:
            str | None: Presented Target result, if any.
        """
        journal = VersionedJournal(
            DiskJournal(path)
        )
        tokens.distribution.return_value = [distribution]
        tokens.loss.return_value = [loss]
        return AttackRun(
            PlainBase("Explain café"),
            target,
            RecordedJudge(
                judge, journal, "Explain café"
            ),
            journal,
            probe=BeastProbe(
                "Explain café",
                ProvidedDesiredResponse(
                    PlainBase("Explain café"),
                    "A café serves coffee"
                ),
                BeamSearch(
                    tokens,
                    ConfiguredBeastBudget(
                        1, 1, 1, 1
                    )
                ),
                journal
            )
        ).run()

    def test_only_actual_target_reply_is_presented_and_assessed(self) -> None:
        with temporary_journal("beast-result.jsonl") as path:
            tokens = Mock(spec=TokenProbabilities)
            tokens.profile.return_value = {"id": "controlled"}
            tokens.decode.return_value = " suffix"
            judge = Mock(spec=Judge)
            judge.description.return_value = {"type": "controlled"}
            judge.assess.return_value = False
            result = self.attempt(
                path, tokens,
                ScriptedChat(["I cannot help with that."]),
                judge, torch.tensor([0.0, 1.0]), 0.25
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert (
                result,
                ParsedJournal(path).read()["assessment"],
                [event["prompt"] for event in events if event.get("role") == "target"], [event["score"] for event in events if event.get("action") == "beast_selection"], [event["text"] for event in events if event["event"] == "result"], judge.assess.call_args.args
            ) == ("I cannot help with that.", False, ["Explain café suffix"], [0.25], ["I cannot help with that."], ("I cannot help with that.",)), "Search evidence was treated as the presented result"

    def test_exhausted_search_has_no_result_or_target_call(self) -> None:
        with temporary_journal("beast-exhausted.jsonl") as path:
            tokens = Mock(spec=TokenProbabilities)
            tokens.profile.return_value = {"id": "controlled"}
            tokens.decode.return_value = ""
            judge = Mock(spec=Judge)
            judge.description.return_value = {"type": "controlled"}
            result = self.attempt(
                path, tokens,
                ScriptedChat([]),
                judge, torch.empty(0), 0.25
            )
            assert (
                result,
                ParsedJournal(path).read()["reason"],
                [event for event in ParsedJournal(path).events() if event.get("role") == "target"]
            ) == (None, "search_exhausted", []), "No-candidate search contacted Target or invented a result"

    def test_preparation_error_remains_failure(self) -> None:
        with temporary_journal("beast-error.jsonl") as path:
            tokens = Mock(spec=TokenProbabilities)
            tokens.profile.return_value = {"id": "controlled"}
            judge = Mock(spec=Judge)
            judge.description.return_value = {"type": "controlled"}
            journal = VersionedJournal(
                DiskJournal(path)
            )
            run = AttackRun(
                PlainBase("Explain café"),
                ScriptedChat([]),
                RecordedJudge(
                    judge, journal, "Explain café"
                ),
                journal,
                probe=BeastProbe(
                    "Explain café",
                    ProvidedDesiredResponse(
                        PlainBase("Explain café"),
                        " "
                    ),
                    BeamSearch(
                        tokens,
                        ConfiguredBeastBudget(
                            1, 1, 1, 1
                        )
                    ),
                    journal
                )
            )
            with self.assertRaises(ValueError):
                _ = run.run()
            assert ParsedJournal(path).read()["status"] == "failure", "Empty answer was silently treated as exhaustion"

    def test_standalone_command_runs_supplied_and_deferred_answers(self) -> None:
        results: list[object] = []
        for option in (("--desired-response", "A café serves coffee"), ("--deferred",)):
            with temporary_journal("beast-cli.jsonl") as path:
                script = (
                    "import main.beast_cli as cli; "
                    "from unittest.mock import Mock; "
                    "from target.model import Model; "
                    "from target.token_probabilities import TokenProbabilities; "
                    "import torch; "
                    "model = Mock(spec=Model); tokens = Mock(spec=TokenProbabilities); "
                    "model.settings.return_value = {'id': 'Qwen/Qwen2.5-0.5B-Instruct', 'revision': '7ae557604adf67be50417f59c2c2f167def9a775'}; "
                    "model.generate.side_effect = lambda messages: 'NO' if messages[0]['role'] == 'system' else 'A café serves coffee'; "
                    "tokens.profile.return_value = model.settings(); tokens.decode.return_value = ' suffix'; "
                    "tokens.distribution.return_value = [torch.tensor([0.0, 1.0])]; "
                    "tokens.loss.return_value = [0.25]; "
                    "cli.snapshot = lambda offline: None; cli.cuda_device = lambda gpu: 'controlled'; "
                     "cli.LazyQwenModel = lambda device, count: model; cli.QwenTokenProbabilities = lambda assets, device: tokens; "
                    "cli.main()"
                )
                subprocess.run(
                    [
                        sys.executable, "-c", script,
                        str(path),
                        "Explain café", "--gpu", "0", "--beast-beams", "1", "--beast-samples", "1", "--beast-suffix-len", "1", *option
                    ],
                    check=True, capture_output=True, text=True
                )
                results.append(
                    ParsedJournal(path).read()["calls"]
                )
        assert results == [
            {
                "target": 1,
                "attacker": 0,
                "judge": 1
            },
            {
                "target": 1,
                "attacker": 1,
                "judge": 1
            }
        ], "The standalone command lost its answer source or skipped final assessment"
