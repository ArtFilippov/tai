"""Common contract for local Qwen failures."""

from abc import ABC, abstractmethod


class QwenError(RuntimeError, ABC):
    """A local preparation, device, or model contract failure without fallback."""

    @abstractmethod
    def category(self) -> str:
        """Identify which stage rejected the local request.

        Returns:
            str: Failure category.
        """
