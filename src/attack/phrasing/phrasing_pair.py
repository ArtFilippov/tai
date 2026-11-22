"""Independent recorded outcomes for both tense experiments."""

from collections.abc import Mapping
from typing import Protocol


class PhrasingPair(Protocol):
    """Run past then future with separate failure boundaries and results."""

    def run(self) -> Mapping[str, Mapping[str, object]]:
        ...

    def description(self) -> dict[str, object]:
        ...
