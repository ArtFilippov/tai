"""Recovery of facts already committed to a run journal."""

import json
from pathlib import Path
from typing import override

from fitd.journal_view import JournalView
from fitd.run_outcome import RunOutcome


class ParsedJournal(JournalView):
    """Reads a finished run without choosing a result or judging its text."""

    def __init__(self, path: Path) -> None:
        """Keep the file to recover.

        Args:
            path (Path): Finished run journal.
        """
        super().__init__()
        self.path = path

    @override
    def read(self) -> RunOutcome:
        text: str | None = None
        assessment: bool | None = None
        status: str | None = None
        calls = {
            "target": 0,
            "attacker": 0,
            "judge": 0
        }
        started = False
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if status is not None:
                    raise ValueError("Events after run completion")
                match json.loads(line):
                    case {"event": "start", "base": str()} if not started:
                        started = True
                    case {"event": "call", "role": "target" | "attacker" | "judge" as role, "prompt": str(), "response": str()} if started:
                        calls[role] += 1
                    case {"event": "step", "prompt": str(), "response": str(), "accepted": bool()} if started:
                        pass
                    case {"event": "result", "text": str() as value} if started and text is None:
                        text = value
                    case {"event": "judgment", "text": str() as value, "assessment": bool() as verdict} if text is not None and value == text and assessment is None:
                        assessment = verdict
                    case {"event": "end", "status": "completed"} if text is not None and assessment is not None:
                        status = "completed"
                    case {"event": "end", "status": "no_result"} if text is None and assessment is None and started:
                        status = "no_result"
                    case _:
                        raise ValueError("Invalid run journal event")
        if status is None:
            raise ValueError("Run journal has no outcome")
        return {
            "status": status,
            "text": text,
            "assessment": assessment,
            "calls": calls
        }
