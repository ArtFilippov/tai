"""Run journal recording boundary."""

from typing import Protocol


class Journal(Protocol):
    """An append-only record of events chosen by the participants."""

    def schema(self) -> dict[str, str | int]:
        ...

    def record(self, event: dict[str, object]) -> None:
        ...
