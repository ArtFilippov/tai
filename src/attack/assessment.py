"""Assessment callback used when FITD selects a Target result."""

from typing import Protocol


class Assessment(Protocol):
    """Accepts a selected result and reports the linked verdict."""

    def __call__(self, text: str) -> bool:
        ...
