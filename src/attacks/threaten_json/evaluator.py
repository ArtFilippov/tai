"""Gate substantive answer assessment on strict JSON parseability."""

import json
from dataclasses import dataclass
from typing import override

from core.evaluation import Evaluator
from core.primitives.messages import Message


def parseable(message: Message) -> bool:
    def reject(value: str) -> None:
        raise ValueError(value)

    try:
        json.loads(message.to_string(), parse_constant=reject)
    except (ValueError, RecursionError):
        return False
    return True


@dataclass(frozen=True)
class JSONAnswerEvaluator(Evaluator):
    """Assess the presented JSON only when it parses as standard JSON.

    Attributes:
        answer (Evaluator): Substantive answer assessment.
    """

    answer: Evaluator

    @override
    def evaluate(self, base: Message, result: Message) -> bool:
        return parseable(result) and self.answer.evaluate(base, result)
