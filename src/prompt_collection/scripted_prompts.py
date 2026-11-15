"""Small source-independent prompt collection."""

from collections.abc import Iterator, Sequence
from typing import override

from prompt_collection.prompt_candidate import PromptCandidate
from prompt_collection.prompt_collection import PromptCollection


class ScriptedPrompts(PromptCollection):
    """Expose supplied candidates and an optional trailing source failure."""

    def __init__(self, prompts: Sequence[PromptCandidate], *, broken: bool = False) -> None:
        """Bind an offline collection with an optional iteration failure.

        Args:
            prompts (Sequence[PromptCandidate]): Ordered candidates.
            broken (bool, default=False): Fail after yielding the supplied prefix.
        """
        super().__init__()
        self.prompts = prompts
        self.broken = broken

    @override
    def description(self) -> dict[str, object]:
        return {"type": "scripted_prompts"}

    @override
    def candidates(self) -> Iterator[PromptCandidate]:
        yield from self.prompts
        if self.broken:
            raise ValueError("Collection unavailable")
