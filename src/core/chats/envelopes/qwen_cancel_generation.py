"""Per-chat cancellation callback passed to local model decoding."""

from dataclasses import dataclass
from threading import Event
from typing import override

from ..qwen_stopping_criterion import StoppingCriterion


@dataclass(frozen=True)
class CancelGeneration(StoppingCriterion):
    """Observe only this request's cancellation signal between decoding steps.

    Attributes:
        cancellation (Event): Signal belonging to one model request.
    """

    cancellation: Event

    def cancelled(self) -> bool:
        return self.cancellation.is_set()

    @override
    def __call__(self, *args: object, **kwargs: object) -> bool:
        return self.cancelled()
