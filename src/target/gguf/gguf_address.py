"""Contract for a selected GGUF service address."""

from typing import Protocol


class GGUFAddress(Protocol):
    """Provide a bind address and its HTTP origin."""

    def host(self) -> str:
        ...

    def port(self) -> int:
        ...

    def origin(self) -> str:
        ...
