"""Generation callback checked between local decoding steps."""

from typing import Protocol


class StoppingCriterion(Protocol):
    """Observe a single conversation's cancellation signal."""

    def __call__(self, *args: object, **kwargs: object) -> bool: ...
