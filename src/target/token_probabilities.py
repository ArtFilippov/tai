"""Backend-independent token distributions and desired-answer loss."""

from collections.abc import Sequence
from typing import Protocol

import torch


class TokenProbabilities(Protocol):
    """Supply distributions, answer losses and their token vocabulary."""

    def distribution(self, requests: Sequence[str]) -> list[torch.Tensor]:
        ...

    def loss(self, requests: Sequence[str], answer: str) -> list[float]:
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        ...

    def profile(self) -> dict[str, str]:
        ...
