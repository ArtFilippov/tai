"""Controlled judgment for the runnable example."""

from typing import override

from fitd.judge import Judge


class EqualJudge(Judge):
    """Assesses whether the presented text equals the expected response."""

    def __init__(self, expected: str) -> None:
        """Remember the text to accept.

        Args:
            expected (str): Text that receives a positive verdict.
        """
        super().__init__()
        self.expected = expected

    @override
    def assess(self, text: str) -> bool:
        return text == self.expected
