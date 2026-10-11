"""Offline model construction from a pinned snapshot."""

from typing import Protocol

from .qwen_model import Model


class ModelLoader(Protocol):
    """Load pinned safe weights with the selected dtype and attention path."""

    def from_pretrained(self, path: str, **options: object) -> Model: ...
