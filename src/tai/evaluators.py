"""Evaluators judge a presented Message; the User decides when to stop."""

from dataclasses import dataclass
from typing import Any

from tai.messages import Message


class Evaluator:
    def settings(self) -> dict[str, Any]:
        """Configuration snapshot only; never evaluate a result here."""
        return {"type": f"{type(self).__module__}.{type(self).__qualname__}"}

    def is_success(self, result: Message) -> bool:
        raise NotImplementedError

    def reason(self, result: Message) -> str:
        raise NotImplementedError

    def evidence(self, result: Message) -> list[str]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ContainsTextEvaluator(Evaluator):
    criterion: str

    def __post_init__(self) -> None:
        if not isinstance(self.criterion, str) or not self.criterion:
            raise ValueError("criterion must be a nonempty string")

    def is_success(self, result: Message) -> bool:
        return self.criterion in result.to_string()

    def settings(self) -> dict[str, Any]:
        return {"type": "contains_text", "criterion": self.criterion}

    def reason(self, result: Message) -> str:
        outcome = "found" if self.is_success(result) else "not found"
        return f"Exact case-sensitive criterion {self.criterion!r} {outcome}."

    def evidence(self, result: Message) -> list[str]:
        return [self.criterion] if self.is_success(result) else []
