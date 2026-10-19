"""Boundary for a configured FITD strategy."""

from typing import Protocol

from target.chat import Chat


class Probe(Protocol):
    """Produces a presented Target response from one FITD conversation."""

    def execute(self, base: str, target: Chat) -> str | None:
        ...
