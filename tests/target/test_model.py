"""Offline pinned-profile model for integration of the GA role assembly."""

from collections.abc import Sequence
from typing import override

from target.model import Model


class TestModel(Model):
    """Answer Target and assessment calls without loading local weights."""

    @override
    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        return "YES" if messages[0]["role"] == "system" else "Detailed answer"

    @override
    def revision(self) -> str:
        return "7ae557604adf67be50417f59c2c2f167def9a775"

    @override
    def settings(self) -> dict[str, object]:
        return {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": self.revision()
        }
