"""Evaluators judge a presented Message; the User decides when to stop."""

from dataclasses import dataclass

from tai.messages import Message


class Evaluator:
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

    def reason(self, result: Message) -> str:
        outcome = "found" if self.is_success(result) else "not found"
        return f"Exact case-sensitive criterion {self.criterion!r} {outcome}."

    def evidence(self, result: Message) -> list[str]:
        return [self.criterion] if self.is_success(result) else []
