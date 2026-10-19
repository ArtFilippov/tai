"""Public attack run boundary."""

from typing import Protocol


class Attack(Protocol):
    """Selects a presented text from a supplied base request."""

    def run(self, base: str) -> str | None:
        ...
