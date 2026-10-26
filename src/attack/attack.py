"""Public attack run boundary."""

from typing import Protocol

class Attack(Protocol):
    """Select a presented text for the request bound to this attack."""

    def description(self) -> dict[str, object]:
        ...

    def run(self) -> str | None:
        ...
