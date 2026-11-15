"""One prompt and its stable source identity."""

from typing import Protocol


class PromptCandidate(Protocol):
    """Keep prompt text independent from source identity and traversal order."""

    def text(self) -> str:
        ...

    def identity(self) -> dict[str, str | int]:
        """Describe the original source, independently of traversal position.

        Returns:
            dict[str, str | int]: Source-defined coordinates, excluding traversal position.
        """
        ...
