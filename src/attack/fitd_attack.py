"""Composition of FITD, judgment and a durable run journal."""

from dataclasses import dataclass
from typing import override

from attack.attack import Attack
from attack.probe import Probe
from base.base_request import BaseRequest
from journal.journal import Journal
from judge.judge import Judge
from target.chat import Chat

SUCCESS = "success"


@dataclass(eq=False)
class FITDAttack(Attack):
    """One attack run that owns selection before requesting judgment.

    Attributes:
        target (Chat): Conversation being attacked.
        judge (Judge): Assessment of the presented result.
        journal (Journal): Destination for all run events.
        probe (Probe): Configured FITD strategy.
        failures (list[str]): Assessment failures during this run.
    """

    target: Chat
    judge: Judge
    journal: Journal
    probe: Probe
    failures: list[str]

    def __init__(self, target: Chat, judge: Judge, journal: Journal,
                 probe: Probe) -> None:
        """Keep the independently injected participants and journal.

        Args:
            target (Chat): Conversation being attacked.
            judge (Judge): Assessment of the presented result.
            journal (Journal): Destination for all run events.
            probe (Probe): Configured FITD strategy.
        """
        super().__init__()
        self.target = target
        self.judge = judge
        self.journal = journal
        self.probe = probe
        self.failures: list[str] = []

    @override
    def run(self, base: BaseRequest) -> str | None:
        self.journal.record(
            {
                "event": "start",
                "base": base.text(),
                "base_metadata": dict(
                    base.metadata()
                )
            }
        )
        try:
            outcome = self.probe.execute(
                base.text(), self.target, self.assess
            )
        except Exception as exc:
            if not self.failures:
                self.journal.record(
                    {
                        "event": "execution_error",
                        "error": repr(exc)
                    }
                )
            self.journal.record(
                {
                    "event": "end",
                    "status": "failure",
                    "reason": "assessment_error" if self.failures else "execution_error"
                }
            )
            raise
        self.journal.record(
            {
                "event": "end",
                "status": outcome.get("status"),
                "reason": "early_success" if outcome.get("status") == SUCCESS else outcome.get("status")
            }
        )
        return outcome.get("text")

    def assess(self, text: str) -> bool:
        """Select a result before handing that exact text to the judge.

        Args:
            text (str): Accepted Target response.

        Returns:
            bool: Whether the judge considers it successful.

        Raises:
            Exception: If the judge cannot assess the submitted result.
        """
        self.journal.record(
            {
                "event": "result",
                "text": text
            }
        )
        try:
            return self.judge.assess(text)
        except Exception as exc:
            self.failures.append(
                repr(exc)
            )
            raise
