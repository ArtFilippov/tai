"""Offline pinned-profile model for integration of the GA role assembly."""

from collections.abc import Sequence
from typing import override

from target.model import Model
from target.sampling import Sampling


class TestModel(Model):
    """Answer Target and assessment calls without loading local weights."""

    def __init__(self, answer: str = "Detailed answer", verdict: str = "YES", *,
                 name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        """Bind independently controlled text, fulfillment verdict and identity.

        Args:
            answer (str, default="Detailed answer"): Target response.
            verdict (str, default="YES"): Judgment response.
            name (str, default="Qwen/Qwen2.5-0.5B-Instruct"): Configured Target identity.
        """
        super().__init__()
        self.answer = answer
        self.verdict = verdict
        self.name = name

    @override
    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        return self.verdict if messages[0]["role"] == "system" else self.answer

    @override
    def sample(self, messages: Sequence[dict[str, str]], sampling: Sampling) -> str:
        return self.generate(messages)

    @override
    def decoding(self, sampling: Sampling) -> dict[str, object]:
        return sampling.description()

    @override
    def prefill(self, messages: Sequence[dict[str, str]], prefix: str) -> str:
        return prefix + self.generate(messages)

    @override
    def revision(self) -> str:
        return "7ae557604adf67be50417f59c2c2f167def9a775"

    @override
    def settings(self) -> dict[str, object]:
        return {
            "id": self.name,
            "revision": self.revision()
        }
