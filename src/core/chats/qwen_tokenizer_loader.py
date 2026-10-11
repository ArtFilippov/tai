"""Offline tokenizer construction from a pinned snapshot."""

from typing import Protocol

from .qwen_tokenizer import Tokenizer


class TokenizerLoader(Protocol):
    """Load a tokenizer without fetching or trusting remote code."""

    def from_pretrained(self, path: str, **options: object) -> Tokenizer: ...
