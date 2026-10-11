"""Pinned model token identifiers for generation."""

from typing import Protocol


class GenerationConfig(Protocol):
    """Token IDs copied into each request's generation configuration."""

    @property
    def bos_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | list[int] | None: ...
