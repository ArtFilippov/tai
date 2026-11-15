"""Composition boundary for source-defined prompt candidates."""

from collections.abc import Mapping
from typing import Protocol

from prompt_collection.prompt_candidate import PromptCandidate


class PromptFactory(Protocol):
    """Create a candidate without coupling a collection to its representation."""

    def __call__(self, text: str, identity: Mapping[str, str | int]) -> PromptCandidate:
        ...
