"""Composition of a probe, judgment and a durable run journal."""

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
class AttackRun(Attack):
    """One attack run that owns selection before requesting judgment.

    Attributes:
        base (BaseRequest): Request associated with this run.
        target (Chat): Conversation being attacked.
        judge (Judge): Assessment of the presented result.
        journal (Journal): Destination for all run events.
        probe (Probe): Configured attack strategy.
        failures (list[str]): Assessment failures during this run.
    """

    base: BaseRequest
    target: Chat
    judge: Judge
    journal: Journal
    probe: Probe
    failures: list[str]

    def __init__(self, base: BaseRequest, target: Chat, judge: Judge, journal: Journal,
                  *, probe: Probe) -> None:
        """Keep the independently injected participants and journal.

        Args:
            base (BaseRequest): Request associated with this run.
            target (Chat): Conversation being attacked.
            judge (Judge): Assessment of the presented result.
            journal (Journal): Destination for all run events.
            probe (Probe): Configured FITD strategy.
        """
        super().__init__()
        self.base = base
        self.target = target
        self.judge = judge
        self.journal = journal
        self.probe = probe
        self.failures: list[str] = []

    @override
    def run(self) -> str | None:
        self.journal.record(
            {
                "event": "start",
                "base": self.base.text(),
                "base_metadata": dict(
                    self.base.metadata()
                ),
                "attack": self.description()
            }
        )
        try:
            outcome = self.probe.execute(
                self.base.text(), self.target, self.assess
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
                    "reason": "assessment_error" if self.failures else "preparation_error" if self.probe.kind() in ("slur_completion", "leak_replay", "false_assertion") and isinstance(exc, ValueError) else "execution_error"
                }
            )
            raise
        self.journal.record(
            {
                "event": "end",
                "status": outcome.get("status"),
                "reason": outcome.get(
                    "reason", "early_success" if outcome.get("status") == SUCCESS else outcome.get("status")
                )
            }
        )
        return outcome.get("text")

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": self.probe.kind(),
            "target": self.target.description(),
            "judge": self.judge.description(),
            "probe": self.probe.description()
        }

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
