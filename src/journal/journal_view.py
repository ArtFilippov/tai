"""Read-only boundary for recovered run facts."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Protocol


class JournalView(Protocol):
    """Recovers recorded facts without generating or reassessing them."""

    def read(self) -> Mapping[str, object]:
        ...

    def outcome(self, events: Sequence[dict[str, object]]) -> Mapping[str, object]:
        ...

    def finish(self, event: dict[str, object], text: str | None, assessment: bool | None, error: str | None) -> Mapping[str, object]:
        ...

    def validate(self, status: str, *, reason: str, text: str | None, assessment: bool | None, error: str | None) -> None:
        ...

    def metrics(self, events: Sequence[dict[str, object]]) -> Mapping[str, object]:
        ...

    def duration(self, events: Sequence[dict[str, object]]) -> float | None:
        ...

    def calls(self, events: Iterable[dict[str, object]]) -> dict[str, int]:
        ...

    def events(self) -> Iterator[dict[str, object]]:
        ...
