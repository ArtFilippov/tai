"""Attach selected model settings to a run's initial journal event."""

from dataclasses import dataclass
from typing import override

from journal.journal import Journal

START = "start"


@dataclass(eq=False)
class ConfiguredJournal(Journal):
    """Preserve a run's effective model configuration in its start event.

    Attributes:
        journal (Journal): Destination of all events.
        models (dict[str, object]): Role-specific model metadata.
    """

    journal: Journal
    models: dict[str, object]

    @override
    def record(self, event: dict[str, object]) -> None:
        if event.get("event") == START:
            self.journal.record(
                {
                    **event,
                    "models": self.models
                }
            )
        else:
            self.journal.record(event)
