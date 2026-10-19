"""Recovery of facts already committed to a run journal."""

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import override

from journal.journal_view import JournalView


@dataclass(eq=False)
class ParsedJournal(JournalView):
    """Reads a finished run without choosing a result or judging its text.

    Attributes:
        path (Path): Finished run journal.
    """

    path: Path

    @override
    def read(self) -> Mapping[str, object]:
        events = list(
            self.events()
        )
        calls = self.calls(events)
        text: str | None = None
        assessment: bool | None = None
        status: str | None = None
        for event in events:
            match event:
                case {"event": "result", "text": str() as value} if text is None:
                    text = value
                case {"event": "judgment", "text": str() as value, "assessment": bool() as verdict} if text is not None and value == text and assessment is None:
                    assessment = verdict
                case {"event": "end", "status": str() as ending} if (
                    (ending, text is None, assessment is None)
                    in {("completed", False, False), ("no_result", True, True)}
                ):
                    status = ending
                case {"event": "result" | "judgment" | "end"}:
                    raise ValueError("Invalid run journal event")
                case _:
                    continue
        if status is None:
            raise ValueError("Run journal has no outcome")
        return {
            "status": status,
            "text": text,
            "assessment": assessment,
            "calls": calls
        }

    def calls(self, events: Iterable[object]) -> dict[str, int]:
        """Count calls while checking the surrounding event sequence.

        Args:
            events (Iterable[object]): Events from the finished journal.

        Returns:
            dict[str, int]: Recorded calls by role.

        Raises:
            ValueError: If the event sequence cannot be a finished run.
        """
        counts = {
            "target": 0,
            "attacker": 0,
            "judge": 0
        }
        started = False
        finished = False
        for event in events:
            if finished:
                raise ValueError("Events after run completion")
            match event:
                case {"event": "start", "base": str()} if not started:
                    started = True
                case {"event": "call", "role": str() as role, "prompt": str(), "response": str()} if started and role in counts:
                    counts[role] += 1
                case ({"event": "step", "prompt": str(), "response": str(), "accepted": bool()}
                      | {"event": "result", "text": str()}
                      | {"event": "judgment", "text": str(), "assessment": bool()}) if started:
                    pass
                case {"event": "end", "status": "completed" | "no_result"} if started:
                    finished = True
                case _:
                    raise ValueError(f"Invalid run journal event in {self.path}")
        return counts

    def events(self) -> Iterator[object]:
        """Yield saved events in their original order.

        Yields:
            object: An event decoded from the journal.
        """
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)
