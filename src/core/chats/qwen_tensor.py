"""Tensor shapes and output slices used by local generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Tensor(Protocol):
    """The tensor operations needed to cut generated tokens from a response."""

    @property
    def shape(self) -> Sequence[int]: ...

    def __getitem__(self, index: tuple[int, slice]) -> Tensor: ...
