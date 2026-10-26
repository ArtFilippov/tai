"""History-free fitness scoring boundary independent of conversations."""

from typing import Protocol


class Logits(Protocol):
    """Score the target completion after one prefix and base request."""

    def profile(self) -> dict[str, str]:
        ...

    def score(self, prefix: str, base: str, target: str) -> float:
        ...
