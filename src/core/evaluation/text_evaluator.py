"""Deterministic assessment by a required text fragment."""

from dataclasses import dataclass
from typing import override

from core.primitives.messages import Message

from .evaluator import Evaluator


@dataclass(frozen=True, slots=True)
class ContainsTextEvaluator(Evaluator):
    criterion: str

    def __post_init__(self) -> None:
        def validate(value: object) -> None:
            if not isinstance(value, str) or not value:
                raise ValueError("criterion must be a nonempty string")

        validate(self.criterion)

    @override
    def evaluate(self, base: Message, result: Message) -> bool:
        return self.criterion in result.to_string()
