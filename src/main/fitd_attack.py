"""Composition of FITD, judgment and a durable run journal."""

from dataclasses import dataclass
from typing import override

from fitd.attack import Attack
from fitd.chat import Chat
from fitd.fitd_probe import FITDProbe
from fitd.journal import Journal
from fitd.judge import Judge
from fitd.recorded_judge import RecordedJudge


@dataclass(eq=False)
class FITDAttack(Attack):
    """One attack run that owns selection before requesting judgment.

    Attributes:
        target (Chat): Conversation being attacked.
        attacker (Chat): Conversation constructing FITD turns.
        judge (Judge): Assessment of the presented result.
        journal (Journal): Destination for all run events.
        turns (int): Maximum target exchanges.
    """

    target: Chat
    attacker: Chat
    judge: Judge
    journal: Journal
    turns: int

    def __init__(self, target: Chat, attacker: Chat, judge: Judge,
                 journal: Journal, *, turns: int = 10) -> None:
        """Keep the independently injected participants and journal.

        Args:
            target (Chat): Conversation being attacked.
            attacker (Chat): Conversation constructing FITD turns.
            judge (Judge): Assessment of the presented result.
            journal (Journal): Destination for all run events.
            turns (int, default=10): Maximum target exchanges.
        """
        super().__init__()
        self.target = target
        self.attacker = attacker
        self.judge = judge
        self.journal = journal
        self.turns = turns

    @override
    def run(self, base: str) -> str | None:
        self.journal.record(
            {
                "event": "start",
                "base": base
            }
        )
        probe = FITDProbe(
            base, self.attacker, self.journal, self.turns
        )
        text = probe.present(
            probe.probe(self.target)
        )
        if text is None:
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
        _ = RecordedJudge(
            self.judge, self.journal
        ).assess(text)
        self.journal.record(
            {
                "event": "end",
                "status": "completed"
            }
        )
        return text
