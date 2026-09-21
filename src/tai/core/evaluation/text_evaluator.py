"""Deterministic assessment by a required text fragment."""

from dataclasses import dataclass

from tai.core.primitives.messages import Message

from .evaluator import Evaluator


@dataclass(frozen=True, slots=True)
class ContainsTextEvaluator(Evaluator):
    criterion: str

    def __post_init__(self) -> None:
        if not isinstance(self.criterion, str) or not self.criterion:
            raise ValueError("criterion must be a nonempty string")

    def evaluate(self, base: Message, result: Message) -> bool:
        return self.criterion in result.to_string()
