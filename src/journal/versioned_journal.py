"""Attach the journal implementation's schema to each event."""

from dataclasses import dataclass
from typing import override

from journal.journal import Journal


@dataclass(eq=False)
class VersionedJournal(Journal):
    """Give every record its destination journal's schema identity."""

    journal: Journal

    @override
    def schema(self) -> dict[str, str | int]:
        return self.journal.schema()

    @override
    def record(self, event: dict[str, object]) -> None:
        self.journal.record(
            {
                **event,
                **self.schema()
            }
        )
