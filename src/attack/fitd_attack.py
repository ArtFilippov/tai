"""Composition of FITD, judgment and a durable run journal."""

from dataclasses import dataclass
from typing import override

from attack.attack import FITDAttackRun
from attack.probe import Probe
from fitd.chat import Chat
from fitd.journal import Journal
from fitd.judge import Judge


@dataclass(eq=False)
class FITDAttack(FITDAttackRun):
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
        if (text := self.probe.execute(base, self.target)) is None:
            self.journal.record(
                {
                    "event": "end",
                    "status": "no_result"
                }
            )
            return None
        self.journal.record(
            {
                "event": "result",
                "text": text
            }
        )
        _ = self.judge.assess(text)
        self.journal.record(
            {
                "event": "end",
                "status": "completed"
            }
        )
        return text
