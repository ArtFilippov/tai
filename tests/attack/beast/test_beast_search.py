"""Loss ordering and continued BEAST beam trials."""

import unittest
from unittest.mock import Mock

import torch

from attack.beast.configured_beast_budget import ConfiguredBeastBudget
from attack.beast.beam_search import BeamSearch
from journal.memory_journal import MemoryJournal
from target.token_probabilities import TokenProbabilities


class TestBeastSearch(unittest.TestCase):
    """Check that sampling is bounded and the lowest loss survives."""

    def test_lower_loss_wins_beam_and_global_selection(self) -> None:
        tokens = Mock(spec=TokenProbabilities)
        def decode(suffix: list[int]) -> str:
            return f" {suffix[-1]}" if suffix else ""
        def loss(requests: list[str], answer: str) -> list[float]:
            scores = {
                1: 4.0,
                2: 1.0,
                3: 3.0,
                4: 2.0
            }
            return [scores[int(request[-1])] for request in requests]
        tokens.decode.side_effect = decode
        tokens.distribution.side_effect = [
            [
                torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0])
            ],
            [
                torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
                torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
            ]
        ]
        tokens.loss.side_effect = loss
        search = BeamSearch(
            tokens,
            ConfiguredBeastBudget(
                2, 1, 2, 1
            )
        )
        candidate = search.search(
            "café", "answer",
            MemoryJournal()
        )
        assert candidate is not None and (
            candidate,
            search.score()
        ) == ([2], 1.0), "The beam maximized loss or returned its last continuation"

    def test_next_trial_extends_previous_suffix_past_length_limit(self) -> None:
        tokens = Mock(spec=TokenProbabilities)
        def decode(suffix: list[int]) -> str:
            return "".join(str(token) for token in suffix)
        tokens.decode.side_effect = decode
        distribution = torch.tensor([0.0, 1.0])
        tokens.distribution.side_effect = [[distribution]] * 4
        tokens.loss.side_effect = [[4.0], [3.0], [2.0], [1.0]]
        candidate = BeamSearch(
            tokens,
            ConfiguredBeastBudget(
                1, 1, 2, 2
            )
        ).search(
            "café", "answer",
            MemoryJournal()
        )
        assert (
            candidate,
            [call.args[0] for call in tokens.distribution.call_args_list]
        ) == ([1, 1, 1, 1], [["café"], ["café1"], ["café11"], ["café111"]]), "Trials restarted or truncated the previous suffix"

    def test_worse_continued_trial_does_not_displace_best_search_score(self) -> None:
        tokens = Mock(spec=TokenProbabilities)
        tokens.decode.return_value = " suffix"
        distribution = torch.tensor([0.0, 1.0])
        tokens.distribution.return_value = [distribution]
        tokens.loss.side_effect = [[0.2], [0.9]]
        search = BeamSearch(
            tokens,
            ConfiguredBeastBudget(
                1, 1, 1, 2
            )
        )
        result = search.search(
            "café", "answer",
            MemoryJournal()
        )
        assert (
            result,
            search.score()
        ) == ([1], 0.2), "The final trial displaced the best scored suffix"

    def test_no_sampled_candidate_is_not_a_computation_error(self) -> None:
        tokens = Mock(spec=TokenProbabilities)
        tokens.decode.return_value = ""
        tokens.distribution.return_value = [
            torch.empty(0)
        ]
        assert BeamSearch(tokens, ConfiguredBeastBudget(2, 1, 2, 1)).search("café", "answer", MemoryJournal()) is None, "Exhausted search invented a candidate"

    def test_nonfinite_fitness_is_an_error(self) -> None:
        tokens = Mock(spec=TokenProbabilities)
        tokens.decode.return_value = " suffix"
        distribution = torch.tensor([0.0, 1.0])
        tokens.distribution.return_value = [distribution]
        tokens.loss.return_value = [
            float("nan")
        ]
        with self.assertRaisesRegex(
            ValueError, "finite"
        ):
            _ = BeamSearch(
                tokens,
                ConfiguredBeastBudget(
                    1, 1, 1, 1
                )
            ).search(
                "café", "answer",
                MemoryJournal()
            )
