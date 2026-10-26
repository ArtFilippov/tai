"""Controlled scoring capability for local GA integration tests."""

from typing import override

from target.logits import Logits


class TestLogits(Logits):
    """Rank native mutated descendants above the initial generation."""

    @override
    def profile(self) -> dict[str, str]:
        return {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775"
        }

    @override
    def score(self, prefix: str, base: str, target: str) -> float:
        return 0.0 if "Evolved" in prefix else 1.0
