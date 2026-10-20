"""Deferred source of one history-free model."""

from typing import Protocol

from target.model import Model
from target.qwen_assets import QwenAssets


class ModelFactory(Protocol):
    """Create a model when the first operation requires it."""

    def __call__(self, assets: QwenAssets) -> Model:
        ...
