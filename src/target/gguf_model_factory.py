"""Contract for a model backed by a selected GGUF service."""

from typing import Protocol

from target.gguf_asset import GGUFAsset
from target.gguf_service import GGUFService
from target.model import Model


class GGUFModelFactory(Protocol):
    """Compose a history-free model from a pinned file and service."""

    def __call__(self, file: GGUFAsset, service: GGUFService, tokens: int) -> Model:
        ...
