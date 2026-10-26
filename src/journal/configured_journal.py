"""Attach selected model settings to a run's initial journal event."""

from dataclasses import dataclass
from typing import override

from journal.journal import Journal
from target.model import Model

START = "start"


@dataclass(eq=False)
class ConfiguredJournal(Journal):
    """Preserve a run's effective model configuration in its start event.

    Attributes:
        journal (Journal): Destination of all events.
        target (Model): Target model shared with the judge.
        attacker (Model): Selected Attacker model.
    """

    journal: Journal
    target: Model
    attacker: Model

    @override
    def record(self, event: dict[str, object]) -> None:
        if event.get("event") == START:
            shared = self.target.settings()
            self.journal.record(
                {
                    **event,
                    "models": {
                        "target": shared,
                        "judge": shared,
                        "attacker": self.attacker.settings()
                    }
                }
            )
        else:
            self.journal.record(event)
