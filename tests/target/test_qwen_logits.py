"""Teacher-forced scoring without Target generation or conversation state."""

import unittest
from unittest.mock import Mock

import torch

from target.qwen_assets import QwenAssets
from target.qwen_logits import QwenLogits


class TestQwenLogits(unittest.TestCase):
    """Check native token slicing and nonmutating forward inference."""

    def test_score_does_not_generate_or_mutate_weights(self) -> None:
        assets = Mock(spec=QwenAssets)
        assets.tokenizer.return_value = {"input_ids": torch.tensor([[2]])}
        assets.tokenizer.apply_chat_template.return_value = {
            "input_ids": torch.tensor(
                [
                    [
                        0, 1, 2, 3
                    ]
                ]
            ),
            "attention_mask": torch.ones(
                (
                    1, 4
                )
            )
        }
        weights = torch.nn.Parameter(
            torch.ones(
                (
                    1, 4, 5
                )
            )
        )
        assets.model.return_value.logits = weights
        _ = QwenLogits(
            assets,
            torch.device("cpu")
        ).score(
            "Prefix", "Base", "Desired"
        )
        assert not assets.model.generate.call_count and weights.grad is None and torch.equal(weights, torch.ones((1, 4, 5))), "Fitness scoring generated a reply or changed model weights"

    def test_score_uses_prefix_plus_base_without_a_history(self) -> None:
        assets = Mock(spec=QwenAssets)
        assets.tokenizer.return_value = {"input_ids": torch.tensor([[2]])}
        assets.tokenizer.apply_chat_template.return_value = {
            "input_ids": torch.tensor(
                [
                    [
                        0, 1, 2, 3
                    ]
                ]
            ),
            "attention_mask": torch.ones(
                (
                    1, 4
                )
            )
        }
        assets.model.return_value.logits = torch.ones(
            (
                1, 4, 5
            )
        )
        _ = QwenLogits(
            assets,
            torch.device("cpu")
        ).score(
            "Prefix", "Base", "Desired"
        )
        assert assets.tokenizer.apply_chat_template.call_args.args[0] == [
            {
                "role": "user",
                "content": "Prefix Base"
            },
            {
                "role": "assistant",
                "content": "Desired"
            }
        ], "The fitness prompt used Target history or omitted the base request"

    def test_score_uses_native_completion_logit_slice(self) -> None:
        assets = Mock(spec=QwenAssets)
        assets.tokenizer.return_value = {"input_ids": torch.tensor([[2]])}
        assets.tokenizer.apply_chat_template.return_value = {
            "input_ids": torch.tensor(
                [
                    [
                        0, 1, 2, 3
                    ]
                ]
            ),
            "attention_mask": torch.ones(
                (
                    1, 4
                )
            )
        }
        logits = torch.zeros(
            (
                1, 4, 5
            )
        )
        logits[0, 2, 2] = 10
        assets.model.return_value.logits = logits
        score = QwenLogits(
            assets,
            torch.device("cpu")
        ).score(
            "Prefix", "Base", "Desired"
        )
        assert score < 0.001, "The fitness loss scored a different token slice from native AutoDAN"
