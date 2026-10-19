"""Judgment boundary for a presented text."""

from typing import Protocol


class Judge(Protocol):
    """An independent assessment of the attack's presented result."""

    def assess(self, text: str) -> bool:
        ...
