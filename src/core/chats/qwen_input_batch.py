"""Movable token batches returned by the optional tokenizer."""

from collections.abc import Iterable
from typing import Protocol, Self

from .qwen_tensor import Tensor


class InputBatch(Protocol):
    """Named model input tensors that move together to the selected device."""

    def keys(self) -> Iterable[str]: ...

    def __getitem__(self, key: str) -> Tensor: ...

    def to(self, device: str) -> Self: ...
