"""Controlled model inputs for conversation-isolation tests."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from target.model import Model


@dataclass(eq=False)
class ScriptedModel(Model):
    """Accepts only the specified complete model requests in order.

    Attributes:
        requests (list[list[dict[str, str]]]): Expected requests.
    """

    requests: list[list[dict[str, str]]]

    @override
    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        if list(messages) != self.requests.pop(0):
            raise ValueError("Model received a different conversation")
        return "Reply"

    @override
    def prefill(self, messages: Sequence[dict[str, str]], prefix: str) -> str:
        if list(messages) != self.requests.pop(0):
            raise ValueError("Model received a different conversation")
        return prefix + "Reply"

    @override
    def revision(self) -> str:
        return "scripted"

    @override
    def settings(self) -> dict[str, object]:
        return {
            "id": "scripted",
            "revision": self.revision()
        }
