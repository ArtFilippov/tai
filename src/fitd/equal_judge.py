"""Controlled judgment for the runnable example."""

from dataclasses import dataclass
from typing import override

from fitd.judge import Judge


@dataclass(frozen=True)
class EqualJudge(Judge):
    """Assesses whether the presented text equals the expected response.

    Attributes:
        expected (str): Text that receives a positive verdict.
    """

    expected: str

    @override
    def assess(self, text: str) -> bool:
        return text == self.expected
