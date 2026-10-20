"""Delay constructing a history-free model until its first operation."""

from collections.abc import Sequence
from typing import override

from qraft import cached_property

from target.model import Model
from target.model_factory import ModelFactory
from target.qwen_assets import QwenAssets


class LazyModel(Model):
    """Share one model created by a deferred factory across all requests."""

    def __init__(self, assets: QwenAssets, factory: ModelFactory) -> None:
        """Hold the factory without creating its model.

        Args:
            assets (QwenAssets): Resources held until the model needs them.
            factory (ModelFactory): Deferred source of one model.
        """
        super().__init__()
        self.assets: QwenAssets = assets
        self.factory: ModelFactory = factory

    @cached_property
    def delegate(self) -> Model:
        return self.factory(self.assets)

    @override
    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        return self.delegate.generate(messages)

    @override
    def revision(self) -> str:
        return self.delegate.revision()
