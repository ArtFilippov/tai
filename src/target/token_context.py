"""Tokenize user continuations and final assistant completions."""

from collections.abc import Sequence
from typing import Protocol

import torch


class TokenContext(Protocol):
    """Express chat contexts without exposing a model's tokenizer or template."""

    def user(self, text: str) -> torch.Tensor:
        """Encode an open user turn whose next token continues its text.

        Args:
            text (str): User text so far.

        Returns:
            torch.Tensor: Open user-turn token IDs.
        """
        ...

    def assistant(self, text: str) -> torch.Tensor:
        """Encode the final user turn and assistant generation header.

        Args:
            text (str): Final user text.

        Returns:
            torch.Tensor: Token IDs through the assistant header.
        """
        ...

    def answer(self, text: str) -> torch.Tensor:
        """Encode the desired assistant answer without special tokens.

        Args:
            text (str): Desired answer text.

        Returns:
            torch.Tensor: Answer token IDs without chat markers.
        """
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        ...
