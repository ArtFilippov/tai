"""Validated immutable profile of a local Transformers conversation."""

from collections.abc import Mapping
from typing import Protocol

from .qwen_checkpoint_contract import LocalCheckpointPort


class QwenProfilePort(Protocol):
    """Validated model profile, separate from each LLMChat's mutable dialogue."""

    def checkpoint(self) -> LocalCheckpointPort: ...

    def dtype(self) -> str: ...

    def generation(self) -> Mapping[str, int | float | bool]: ...

    @property
    def system(self) -> str: ...

    def details(self) -> dict[str, object]: ...
