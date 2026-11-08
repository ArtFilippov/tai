"""Batched inference and answer scoring keep their token boundaries."""

import unittest
from unittest.mock import Mock

import torch
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from target.computed_token_probabilities import ComputedTokenProbabilities
from target.token_source import TokenSource
from target.transformers_assets import TransformersAssets
from target.transformers_token_inference import TransformersTokenInference


class TestTokenInference(unittest.TestCase):
    """Check variable-length batches without backend details in the contract."""

    def test_transformers_infers_with_another_causal_model(self) -> None:
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=11, n_embd=16, n_layer=1, n_head=2
            )
        )
        _ = model.eval()
        assets = Mock(spec=TransformersAssets)
        assets.model = model
        first = torch.tensor([1, 2])
        second = torch.tensor([1, 2, 3])
        result = TransformersTokenInference(
            assets,
            torch.device("cpu"),
            {"id": "tiny GPT-2"}
        ).infer([first, second])
        assert [list(logits.shape) for logits in result] == [[2, 11], [3, 11]], "The generic Transformers adapter rejected another causal model"

    def test_transformers_batch_masks_shorter_sequences(self) -> None:
        model = Mock()
        assets = Mock(spec=TransformersAssets)
        assets.model = model
        model.return_value.logits = torch.zeros(
            (
                2, 4, 5
            )
        )
        inference = TransformersTokenInference(
            assets, torch.device("cpu"), {"id": "controlled"}
        )
        result = inference.infer(
            [
                torch.tensor([1, 2]),
                torch.tensor([1, 2, 3, 4])
            ]
        )
        assert (
            [list(logits.shape) for logits in result],
            model.call_args.kwargs["attention_mask"].tolist()
        ) == ([[2, 5], [4, 5]], [[1, 1, 0, 0], [1, 1, 1, 1]]), "The backend lost batch lengths or attended padding"

    def test_batch_loss_scores_each_answer_at_its_own_boundary(self) -> None:
        source = Mock(spec=TokenSource)
        source.answer.return_value = torch.tensor([3])
        source.assistant.side_effect = [
            torch.tensor([1, 2]),
            torch.tensor([1, 2, 4])
        ]
        logits = torch.zeros(
            (
                3, 5
            )
        )
        logits[1, 3] = 10
        second = torch.zeros(
            (
                4, 5
            )
        )
        second[2, 3] = 10
        source.infer.return_value = [logits, second]
        result = ComputedTokenProbabilities(source).loss(
            [
                "first", "second"
            ],
            "answer"
        )
        assert all(score < 0.001 for score in result) and source.infer.call_count == 1, "Batch scoring used the wrong answer boundary or separate calls"
