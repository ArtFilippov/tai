"""Shared, history-free model generation boundary."""

from collections.abc import Sequence
from typing import Protocol


class Model(Protocol):
    """Generates one reply from a complete role-tagged conversation."""

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        ...

    def prefill(self, messages: Sequence[dict[str, str]], prefix: str) -> str:
        """Continue a partial assistant response.

        Args:
            messages (Sequence[dict[str, str]]): Complete request history.
            prefix (str): Assistant text already supplied.

        Returns:
            str: Complete assistant message including prefix.
        """
        ...

    def revision(self) -> str:
        ...

    def settings(self) -> dict[str, object]:
        ...
