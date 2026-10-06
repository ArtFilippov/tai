"""The control is evidence; only the pressured JSON answer is assessed."""

from typing import override

from tai import Evaluator, Message


class AnswerJudge(Evaluator):
    """Record exactly the presented answer passed to the answer rubric."""

    def __init__(self, answer: bool, pairs: list[tuple[Message, Message]]) -> None:
        """Store the scripted assessment.

        Args:
            answer (bool): Substantive verdict.
            pairs (list[tuple[Message, Message]]): Captured base/result pairs.
        """
        super().__init__()
        self.answer = answer
        self.pairs = pairs

    @override
    def evaluate(self, base: Message, result: Message) -> bool:
        self.pairs.append((base, result))
        return self.answer
