"""Shared, history-free model generation boundary."""

from collections.abc import Sequence
from typing import Protocol


class Model(Protocol):
    """Generates one reply from a complete role-tagged conversation."""

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        ...

    def revision(self) -> str:
        ...

    def settings(self) -> dict[str, object]:
        ...
