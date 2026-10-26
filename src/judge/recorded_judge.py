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
    def description(self) -> dict[str, object]:
        return {
            "type": "recorded_judge",
            "judge": self.judge.description()
        }

    @override
    def assess(self, text: str) -> bool:
        try:
            assessment = self.judge.assess(text)
        except Exception as exc:
            self.journal.record(
                {
                    "event": "call",
                    "role": "judge",
                    "base": self.base,
                    "prompt": text,
                    "response": None
                }
            )
            self.journal.record(
                {
                    "event": "assessment_error",
                    "base": self.base,
                    "text": text,
                    "error": repr(exc)
                }
            )
            raise
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
