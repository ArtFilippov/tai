"""Evaluators judge a presented Message; the User decides when to stop."""

from typing import Protocol

from tai.core.primitives.messages import Message


class Evaluator(Protocol):
    """The entire working contract; inheritance is optional."""

    def evaluate(self, base: Message, result: Message) -> bool:
        """Judge the full presented text against this explicit base request."""
        ...


class EvaluationChat(Protocol):
    """Only the chat operations needed by a model classifier."""

    def reset(self) -> None: ...

    def send(self, message: Message) -> Message: ...


def evaluate(evaluator: Evaluator, base: Message, result: Message) -> bool:
    """Enforce the contract at every execution/readiness boundary."""
    success = evaluator.evaluate(base, result)
    if type(success) is not bool:
        raise TypeError("Evaluator.evaluate must return bool")
    return success
