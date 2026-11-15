"""Deterministic shared model for offline attack execution."""

from collections.abc import Sequence
from typing import override

from target.model import Model


class QueuedModel(Model):
    """Consume real model calls in order and fail if execution exceeds the script."""

    def __init__(self, replies: list[str], *, failure: str = "Model unavailable") -> None:
        """Bind the responses and the technical failure after their exhaustion.

        Args:
            replies (list[str]): Target and judge responses in call order.
            failure (str, default="Model unavailable"): Error on an unscripted call.
        """
        super().__init__()
        self.replies = replies
        self.failure = failure

    @override
    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        if not self.replies:
            raise ValueError(self.failure)
        return self.replies.pop(0)

    @override
    def prefill(self, messages: Sequence[dict[str, str]], prefix: str) -> str:
        return prefix + self.generate(messages)

    @override
    def revision(self) -> str:
        return "offline-revision"

    @override
    def settings(self) -> dict[str, object]:
        return {
            "id": "Offline/Queued",
            "revision": "offline-revision",
        }
