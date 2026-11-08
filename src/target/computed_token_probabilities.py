"""Pair a generic token source with computed distributions and losses."""

from collections.abc import Sequence
from typing import override

import torch
from torch.nn import functional

from target.token_probabilities import TokenProbabilities
from target.token_source import TokenSource


class ComputedTokenProbabilities(TokenProbabilities):
    """Expose shared probability calculations over an injected token source."""

    def __init__(self, source: TokenSource) -> None:
        """Keep a compatible inference and token context together.

        Args:
            source (TokenSource): Context and causal inference capability.
        """
        super().__init__()
        self.source = source

    @override
    def profile(self) -> dict[str, str]:
        return self.source.profile()

    @override
    def decode(self, tokens: Sequence[int]) -> str:
        return self.source.decode(tokens)

    @override
    def distribution(self, requests: Sequence[str]) -> list[torch.Tensor]:
        """Return the next-token distribution for each open user turn.

        Args:
            requests (Sequence[str]): Texts to continue as user messages.

        Returns:
            list[torch.Tensor]: One distribution per request, on the model device.
        """
        if not requests:
            return []
        return [
            torch.softmax(logits[-1].float(), dim=-1)
            for logits in self.source.infer(
                [self.source.user(request) for request in requests]
            )
        ]

    @override
    def loss(self, requests: Sequence[str], answer: str) -> list[float]:
        """Mean desired-answer loss for each complete user request.

        Args:
            requests (Sequence[str]): Texts sent to the final Target.
            answer (str): Desired assistant completion.

        Returns:
            list[float]: Mean negative log likelihoods, one per request.

        Raises:
            ValueError: If the desired answer has no tokens.
        """
        if not requests:
            return []
        labels = self.source.answer(answer)
        if labels.numel() < 1:
            raise ValueError("Desired response has no tokens")
        prefixes = [self.source.assistant(request) for request in requests]
        losses: list[float] = []
        for index in range(
            0,
            len(prefixes),
            8
        ):
            batch = prefixes[index:index + 8]
            inputs = [torch.cat((prefix, labels.to(prefix.device))) for prefix in batch]
            logits = self.source.infer(inputs)
            for prefix, output in zip(
                batch, logits, strict=True
            ):
                start = prefix.numel() - 1
                entropy = functional.cross_entropy(
                    output[start:-1].float(),
                    labels.to(output.device)
                )
                losses.append(
                    entropy.item()
                )
        return losses
