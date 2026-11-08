"""Qwen assembly uses the same final chat context as the Target."""

import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch

from target.local_qwen_assets import LocalQwenAssets
from target.qwen_assets import QwenAssets
from target.envelopes.qwen_token_probabilities import QwenTokenProbabilities


class TestQwenTokenProbabilities(unittest.TestCase):
    """Check final request templating and teacher-forced completion loss."""

    def test_assembly_does_not_load_model_or_tokenizer(self) -> None:
        with patch.object(
            LocalQwenAssets, "model", new_callable=PropertyMock
        ) as model, patch.object(
            LocalQwenAssets, "tokenizer", new_callable=PropertyMock
        ) as tokenizer:
            source = QwenTokenProbabilities(
                LocalQwenAssets(
                    torch.device("cpu")
                ),
                torch.device("cpu")
            )
            _ = source.profile()
            assert (model.call_count, tokenizer.call_count) == (0, 0), "Token assembly eagerly loaded Qwen resources"

    def test_scoring_uses_same_user_text_as_final_target_request(self) -> None:
        assets = Mock(spec=QwenAssets)
        assets.tokenizer.apply_chat_template.return_value = torch.tensor(
            [
                [
                    0, 1, 2
                ]
            ]
        )
        assets.tokenizer.return_value = {"input_ids": torch.tensor([[3]])}
        assets.model.return_value.logits = torch.zeros(
            (
                1, 4, 5
            )
        )
        _ = QwenTokenProbabilities(
            assets,
            torch.device("cpu")
        ).loss(
            ["Explain café suffix"], "response"
        )
        assert assets.tokenizer.apply_chat_template.call_args.args[0] == [
            {
                "role": "user",
                "content": "Explain café suffix"
            }
        ], "Search and final Target contexts did not match"

    def test_lower_completion_loss_favors_the_desired_token(self) -> None:
        assets = Mock(spec=QwenAssets)
        assets.tokenizer.apply_chat_template.return_value = torch.tensor(
            [
                [
                    0, 1, 2
                ]
            ]
        )
        assets.tokenizer.return_value = {"input_ids": torch.tensor([[3]])}
        logits = torch.zeros(
            (
                1, 4, 5
            )
        )
        logits[0, 2, 3] = 10
        assets.model.return_value.logits = logits
        score = QwenTokenProbabilities(
            assets,
            torch.device("cpu")
        ).loss(
            ["Explain café suffix"], "response"
        )
        assert score[0] < 0.001, "BEAST scored the wrong desired-answer token or reversed loss direction"

    def test_suffix_distribution_keeps_user_turn_open(self) -> None:
        assets = Mock(spec=QwenAssets)
        assets.tokenizer.apply_chat_template.return_value = torch.tensor(
            [
                [
                    0, 1, 2
                ]
            ]
        )
        assets.model.return_value.logits = torch.zeros(
            (
                1, 3, 5
            )
        )
        _ = QwenTokenProbabilities(
            assets,
            torch.device("cpu")
        ).distribution(["Explain café suffix"])
        assert assets.tokenizer.apply_chat_template.call_args.kwargs["continue_final_message"] is True, "Suffix distribution used the assistant turn"
