"""Read boundary for an in-memory FITD run observation."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from journal.journal import Journal


class Observation(Journal, Protocol):
    """Expose recorded events and candidate measurements for one run."""

    def events(self) -> Sequence[dict[str, object]]:
        ...

    def report(self) -> Mapping[str, object]:
        ...

    def candidate(self, prompt: str) -> bool:
        ...

    def changes(self, base: str) -> list[dict[str, object]]:
        ...

    def requests(self) -> list[str]:
        ...

    def outcome(self) -> Mapping[str, object]:
        ...
