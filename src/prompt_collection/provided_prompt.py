"""A supplied prompt with source-defined identity."""

from collections.abc import Mapping
from typing import override

from prompt_collection.prompt_candidate import PromptCandidate


class ProvidedPrompt(PromptCandidate):
    """Preserve supplied text and source coordinates independently of its use."""

    def __init__(self, text: str, identity: Mapping[str, str | int]) -> None:
        """Snapshot the supplied prompt and its source identity.

        Args:
            text (str): Unchanged rendered prompt.
            identity (Mapping[str, str | int]): Coordinates defined by the source.
        """
        super().__init__()
        self.prompt = text
        self.origin = dict(identity)

    @override
    def text(self) -> str:
        return self.prompt

    @override
    def identity(self) -> dict[str, str | int]:
        return dict(self.origin)
