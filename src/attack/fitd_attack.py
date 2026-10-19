"""Composition of FITD, judgment and a durable run journal."""

from dataclasses import dataclass
from typing import override

from attack.attack import Attack
from attack.probe import Probe
from journal.journal import Journal
from judge.judge import Judge
from target.chat import Chat


@dataclass(eq=False)
class FITDAttack(Attack):
    """One attack run that owns selection before requesting judgment.

    Attributes:
        target (Chat): Conversation being attacked.
        judge (Judge): Assessment of the presented result.
        journal (Journal): Destination for all run events.
        probe (Probe): Configured FITD strategy.
    """

    target: Chat
    judge: Judge
    journal: Journal
    probe: Probe

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

    @override
    def run(self, base: str) -> str | None:
        self.journal.record(
            {
                "event": "start",
                "base": base
            }
        )
        outcome = self.probe.execute(
            base, self.target, self.assess
        )
        self.journal.record(
            {
                "event": "end",
                "status": outcome.get("status")
            }
        )
        return outcome.get("text")

    def assess(self, text: str) -> bool:
        """Select a result before handing that exact text to the judge.

        Args:
            text (str): Accepted Target response.

        Returns:
            bool: Whether the judge considers it successful.
        """
        self.journal.record(
            {
                "event": "result",
                "text": text
            }
        )
        return self.judge.assess(text)
