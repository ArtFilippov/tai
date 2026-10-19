"""Read-only boundary for recovered run facts."""

from collections.abc import Mapping
from typing import Protocol


class JournalView(Protocol):
    """Recovers recorded facts without generating or reassessing them."""

    def read(self) -> Mapping[str, object]:
        ...
