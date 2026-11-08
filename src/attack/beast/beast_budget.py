"""Explicit finite bounds for a request-bound BEAST search."""

from typing import Protocol


class BeastBudget(Protocol):
    """Expose the four finite dimensions of a BEAST search.

    """

    def beams(self) -> int:
        ...

    def samples(self) -> int:
        ...

    def suffix_len(self) -> int:
        ...

    def trials(self) -> int:
        ...

    def validate(self) -> None:
        """Reject unbounded or impossible search dimensions."""
        ...
