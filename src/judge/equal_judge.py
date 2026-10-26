"""Controlled judgment for the runnable example."""

from dataclasses import dataclass
from typing import override

from judge.judge import Judge


@dataclass(frozen=True)
class EqualJudge(Judge):
    """Assesses whether the presented text equals the expected response.

    Attributes:
        expected (str): Text that receives a positive verdict.
    """

    expected: str

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "equal_judge",
            "expected": self.expected
        }

    @override
    def assess(self, text: str) -> bool:
        return text == self.expected
