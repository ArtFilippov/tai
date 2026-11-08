"""Pair causal inference with its matching token context."""

from collections.abc import Sequence
from typing import override

import torch

from target.token_context import TokenContext
from target.token_inference import TokenInference
from target.token_source import TokenSource


class PairedTokenSource(TokenSource):
    """Keep token IDs and logits from compatible injected capabilities."""

    def __init__(self, inference: TokenInference, context: TokenContext) -> None:
        """Bind one context to its matching causal inference backend.

        Args:
            inference (TokenInference): Model returning causal logits.
            context (TokenContext): Tokenizer and chat template for the same model.
        """
        super().__init__()
        self.inference = inference
        self.context = context

    @override
    def infer(self, sequences: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return self.inference.infer(sequences)

    @override
    def profile(self) -> dict[str, str]:
        return self.inference.profile()

    @override
    def user(self, text: str) -> torch.Tensor:
        return self.context.user(text)

    @override
    def assistant(self, text: str) -> torch.Tensor:
        return self.context.assistant(text)

    @override
    def answer(self, text: str) -> torch.Tensor:
        return self.context.answer(text)

    @override
    def decode(self, tokens: Sequence[int]) -> str:
        return self.context.decode(tokens)
