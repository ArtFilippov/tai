"""Ordered prompt traversal independent of its consumers."""

from collections.abc import Iterator
from typing import Protocol

from prompt_collection.prompt_candidate import PromptCandidate


class PromptCollection(Protocol):
    """Supply rendered candidates lazily without hiding source failures."""

    def description(self) -> dict[str, object]:
        """Describe the configured prompt source.

        Returns:
            dict[str, object]: Source configuration and provenance.
        """
        ...

    def candidates(self) -> Iterator[PromptCandidate]:
        """Traverse rendered prompts in collection order, propagating source errors.

        Returns:
            Iterator[PromptCandidate]: Prompt texts with stable source coordinates.
        """
        ...
