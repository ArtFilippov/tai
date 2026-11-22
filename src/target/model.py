"""Shared, history-free model generation boundary."""

from collections.abc import Sequence
from typing import Protocol

from target.sampling import Sampling


class Model(Protocol):
    """Generates one reply from a complete role-tagged conversation."""

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        ...

    def sample(self, messages: Sequence[dict[str, str]], sampling: Sampling) -> str:
        """Generate under an explicit run-local stochastic policy.

        Args:
            messages (Sequence[dict[str, str]]): Complete request history.
            sampling (Sampling): Bounded decoding and isolated randomness.

        Returns:
            str: Complete decoded response.
        """
        ...

    def decoding(self, sampling: Sampling) -> dict[str, object]:
        """Describe effective stochastic decoding, including backend defaults.

        Args:
            sampling (Sampling): Per-run overrides and random-stream policy.

        Returns:
            dict[str, object]: Effective backend-specific generation settings.
        """
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
