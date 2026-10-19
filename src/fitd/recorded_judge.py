"""The judge records the result it received and its assessment."""

from dataclasses import dataclass
from typing import override

from fitd.journal import Journal
from fitd.judge import Judge


@dataclass(eq=False)
class RecordedJudge(Judge):
    """A judge whose assessment is durably linked to its input text.

    Attributes:
        judge (Judge): Source of the verdict.
        journal (Journal): Destination for the linked verdict.
    """

    judge: Judge
    journal: Journal

    @override
    def assess(self, text: str) -> bool:
        assessment = self.judge.assess(text)
        self.journal.record(
            {
                "event": "call",
                "role": "judge",
                "prompt": text,
                "response": str(assessment)
            }
        )
        self.journal.record(
            {
                "event": "judgment",
                "text": text,
                "assessment": assessment
            }
        )
        return assessment
