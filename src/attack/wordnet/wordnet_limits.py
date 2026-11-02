"""Read-only bounds supplied to lexical traversal."""

from typing import Protocol


class WordnetLimits(Protocol):
    """Expose the maximum work and deterministic seed of one exploration."""

    def nodes(self) -> int:
        ...

    def calls(self) -> int:
        ...

    def seed(self) -> int:
        ...

    def validate(self) -> None:
        ...
