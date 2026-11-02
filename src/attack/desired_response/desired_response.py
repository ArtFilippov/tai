"""Request-bound answer source contract."""

from typing import Protocol

from journal.journal import Journal


class DesiredResponse(Protocol):
    """Supply a prepared answer for the exact selected base request."""

    def response(self, base: str, journal: Journal) -> str:
        ...

    def description(self) -> dict[str, object]:
        ...
