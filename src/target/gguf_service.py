"""Context contract for a verified local GGUF service."""

from contextlib import AbstractContextManager
from typing import Protocol


class GGUFService(Protocol):
    """Report ownership while keeping one verified service available."""

    def open(self) -> AbstractContextManager[bool]:
        ...

    def address(self) -> str:
        ...

    def settings(self) -> dict[str, object]:
        ...
