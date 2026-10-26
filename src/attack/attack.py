"""Public attack run boundary."""

from typing import Protocol

from base.base_request import BaseRequest


class Attack(Protocol):
    """Selects a presented text from a supplied base request."""

    def description(self) -> dict[str, object]:
        ...

    def run(self, base: BaseRequest) -> str | None:
        ...
