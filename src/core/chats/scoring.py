"""Optional history-free local continuation scoring capability."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class ContinuationScorer(Protocol):
    @abstractmethod
    def score_continuation(self, prompt: str, expected: str) -> float:
        """Return mean token cross-entropy without changing a conversation.

        Args:
            prompt (str): The entire user prompt.
            expected (str): Assistant continuation to score.

        Returns:
            float: Mean cross-entropy of the continuation.

        Raises:
            NotImplementedError: An implementation must supply a local scorer.
        """
        raise NotImplementedError


def require_scorer(target: object) -> ContinuationScorer:
    if not isinstance(target, ContinuationScorer):
        raise ValueError("AutoDAN requires a local Target with logits continuation scoring (backend=qwen)")
    return target
