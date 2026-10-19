"""The judge records the result it received and its assessment."""

from typing import override

from fitd.journal import Journal
from fitd.judge import Judge


class RecordedJudge(Judge):
    """A judge whose assessment is durably linked to its input text."""

    def __init__(self, judge: Judge, journal: Journal) -> None:
        """Keep the assessment delegate and the run's journal.

        Args:
            judge (Judge): Source of the verdict.
            journal (Journal): Destination for the linked verdict.
        """
        super().__init__()
        self.judge = judge
        self.journal = journal

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
