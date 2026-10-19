"""The judge records the result it received and its assessment."""

from dataclasses import dataclass
from typing import override

from journal.journal import Journal
from judge.judge import Judge


@dataclass(eq=False)
class RecordedJudge(Judge):
    """A judge whose assessment is durably linked to its input text.

    Attributes:
        judge (Judge): Source of the verdict.
        journal (Journal): Destination for the linked verdict.
        base (str): Base request associated with each verdict.
    """

    judge: Judge
    journal: Journal
    base: str

    @override
    def assess(self, text: str) -> bool:
        assessment = self.judge.assess(text)
        self.journal.record(
            {
                "event": "call",
                "role": "judge",
                "base": self.base,
                "prompt": text,
                "response": str(assessment)
            }
        )
        self.journal.record(
            {
                "event": "judgment",
                "base": self.base,
                "text": text,
                "assessment": assessment
            }
        )
        return assessment
