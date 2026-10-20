"""Deferred model creation must preserve one shared instance."""

import unittest

import torch

from target.lazy_model import LazyModel
from target.local_qwen_assets import LocalQwenAssets
from target.scripted_model import ScriptedModel


class TestLazyModel(unittest.TestCase):
    """Check when a model is built and whether subsequent calls reuse it."""

    def test_construction_does_not_call_factory(self) -> None:
        script = ScriptedModel([])
        models = [script]
        _ = LazyModel(
            LocalQwenAssets(
                torch.device("cpu")
            ),
            lambda assets: models.pop()
        )
        assert len(models) == 1, "The model was built before it was needed"

    def test_generations_share_one_model(self) -> None:
        message = {
            "role": "user",
            "content": "One"
        }
        messages = [message]
        script = ScriptedModel([messages, messages])
        models = [script]
        model = LazyModel(
            LocalQwenAssets(
                torch.device("cpu")
            ),
            lambda assets: models.pop()
        )
        _ = model.generate(messages)
        response = model.generate(messages)
        assert response == "Reply", "A repeated request did not use the same model"

    def test_revision_and_generation_share_one_model(self) -> None:
        message = {
            "role": "user",
            "content": "One"
        }
        messages = [message]
        script = ScriptedModel([messages])
        models = [script]
        model = LazyModel(
            LocalQwenAssets(
                torch.device("cpu")
            ),
            lambda assets: models.pop()
        )
        _ = model.revision()
        assert model.generate(messages) == "Reply", "Revision loaded a different model"
