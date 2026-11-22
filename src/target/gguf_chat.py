"""Explicit HTTP completion boundary for shared GGUF generation."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from target.model import Model


class GGUFChat(Model, Protocol):
    """Preserve model operations while describing and validating HTTP completions."""

    def policy(self) -> dict[str, int]:
        ...

    def completion(self, messages: Sequence[dict[str, str]]) -> object:
        ...

    def request(self, messages: Sequence[dict[str, str]], policy: Mapping[str, int | float]) -> object:
        ...

    def response(self, payload: object) -> str:
        ...
