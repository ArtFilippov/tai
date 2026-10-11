"""Context capacity of a locally loaded model."""

from typing import Protocol


class ModelConfig(Protocol):
    """Maximum context length before a complete request would overflow."""

    @property
    def max_position_embeddings(self) -> int: ...
