"""Chat templating for compatible Transformers causal tokenizers."""

from collections.abc import Sequence
from typing import Any, override

import torch
from target.token_context import TokenContext
from target.transformers_assets import TransformersAssets


class TransformersTokenContext(TokenContext):
    """Keep user suffixes open and score answers after the assistant header."""

    def __init__(self, assets: TransformersAssets, device: torch.device) -> None:
        """Keep any compatible chat tokenizer lazy until use.

        Args:
            assets (TransformersAssets): Deferred tokenizer and model.
            device (torch.device): Device for the encoded token IDs.
        """
        super().__init__()
        self.assets = assets
        self.device = device

    @override
    def user(self, text: str) -> torch.Tensor:
        tokenizer: Any = self.assets.tokenizer
        return tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": text
                }
            ],
            tokenize=True, continue_final_message=True, return_tensors="pt"
        )[0].to(self.device)

    @override
    def assistant(self, text: str) -> torch.Tensor:
        tokenizer: Any = self.assets.tokenizer
        return tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": text
                }
            ],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )[0].to(self.device)

    @override
    def answer(self, text: str) -> torch.Tensor:
        tokenizer: Any = self.assets.tokenizer
        return tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].to(self.device)

    @override
    def decode(self, tokens: Sequence[int]) -> str:
        tokenizer: Any = self.assets.tokenizer
        return tokenizer.decode(
            list(tokens),
            skip_special_tokens=False
        )
