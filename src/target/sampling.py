"""Per-run stochastic decoding without changing a shared model."""

from contextlib import AbstractContextManager
from typing import Protocol

import torch


class Sampling(Protocol):
    """Own a bounded decoding policy and its advancing isolated random streams."""

    def options(self) -> dict[str, int | float | bool]:
        ...

    def scope(self, device: torch.device) -> AbstractContextManager[None]:
        ...

    def draw(self) -> int:
        ...

    def description(self) -> dict[str, object]:
        ...
