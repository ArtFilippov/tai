"""Contract for a request-bound BEAST beam search."""

from typing import Protocol

from journal.journal import Journal


class BeastSearch(Protocol):
    """Return a suffix and expose its independent search evidence."""

    def sample(self, base: str, suffixes: list[list[int]], count: int) -> list[list[int]]:
        ...

    def search(self, base: str, answer: str, journal: Journal) -> list[int] | None:
        ...

    def score(self) -> float:
        ...

    def decode(self, suffix: list[int]) -> str:
        ...

    def description(self) -> dict[str, object]:
        ...
