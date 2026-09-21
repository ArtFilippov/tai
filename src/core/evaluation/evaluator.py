"""Evaluators judge a presented Message; the User decides when to stop."""

from abc import abstractmethod
from typing import Protocol

from core.primitives.messages import Message


class Evaluator(Protocol):
    """The entire working contract; inheritance is optional."""

    @abstractmethod
    def evaluate(self, base: Message, result: Message) -> bool:
        """Judge the full presented text against this explicit base request.

        Args:
            base (Message): Explicit original request.
            result (Message): Complete presented result.

        Returns:
            bool: Whether the presented result succeeds.

        Raises:
            NotImplementedError: A concrete evaluator must implement assessment.
        """
        raise NotImplementedError


class EvaluationChat(Protocol):
    """Only the chat operations needed by a model classifier."""

    def reset(self) -> None: ...

    @abstractmethod
    def send(self, message: Message) -> Message:
        raise NotImplementedError


def require_boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("Evaluator.evaluate must return bool")
    return value


def evaluate(evaluator: Evaluator, base: Message, result: Message) -> bool:
    """Enforce the contract at every execution/readiness boundary.

    Args:
        evaluator (Evaluator): Assessment implementation.
        base (Message): Explicit original request.
        result (Message): Complete presented result.

    Returns:
        bool: Strict boolean assessment.
    """
    return require_boolean(evaluator.evaluate(base, result))
