"""Deterministic assessment by a required text fragment."""

from dataclasses import dataclass
from typing import override

from core.primitives.messages import Message
from core.primitives.messages.validation import require_nonempty_text

from .evaluator import Evaluator


@dataclass(frozen=True, slots=True)
class ContainsTextEvaluator(Evaluator):
    criterion: str

    def __post_init__(self) -> None:
        _ = require_nonempty_text(self.criterion, "criterion")

    @override
    def evaluate(self, base: Message, result: Message) -> bool:
        return self.criterion in result.to_string()
