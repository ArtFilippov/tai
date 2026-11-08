"""Backend-independent batched causal language-model inference."""

from collections.abc import Sequence
from typing import Protocol

import torch


class TokenInference(Protocol):
    """Return next-token logits at every position of each input sequence."""

    def infer(self, sequences: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Keep logits on the inference device, one matrix per input.

        Args:
            sequences (Sequence[torch.Tensor]): Unpadded one-dimensional token IDs.

        Returns:
            list[torch.Tensor]: Logits for each position of each input.
        """
        ...

    def profile(self) -> dict[str, str]:
        ...
