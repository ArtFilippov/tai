"""Forward token distributions and losses from a composed source."""

from collections.abc import Sequence
from typing import override

import torch

from target.token_probabilities import TokenProbabilities


class TokenProbabilitiesEnvelope(TokenProbabilities):
    """Preserve the probability interface around an injected delegate."""

    def __init__(self, delegate: TokenProbabilities) -> None:
        """Store the shared token probability delegate.

        Args:
            delegate (TokenProbabilities): Probability source to forward to.
        """
        super().__init__()
        self.delegate = delegate

    @override
    def distribution(self, requests: Sequence[str]) -> list[torch.Tensor]:
        return self.delegate.distribution(requests)

    @override
    def loss(self, requests: Sequence[str], answer: str) -> list[float]:
        return self.delegate.loss(
            requests, answer
        )

    @override
    def decode(self, tokens: Sequence[int]) -> str:
        return self.delegate.decode(tokens)

    @override
    def profile(self) -> dict[str, str]:
        return self.delegate.profile()
