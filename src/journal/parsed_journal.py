"""Recovery of facts already committed to a run journal."""

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import override

from journal.journal_view import JournalView

COMPLETED = "completed"
SUCCESS = "success"
ROUND_LIMIT = "round_limit"
NO_RESULT = "no_result"
FAILURE = "failure"
REASON = "reason"
EXECUTION_ERROR = "execution_error"
SERIES = "series"
START = "start"
SCHEMA_ID = "tai.run"


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
        metrics = self.metrics(events)
        return {
            **self.outcome(events),
            **metrics,
            "duration": self.duration(events)
        }

    @override
    def outcome(self, events: Sequence[dict[str, object]]) -> Mapping[str, object]:
        """Recover the final result and its linked assessment.

        Args:
            events (Sequence[dict[str, object]]): Saved events in order.

        Returns:
            Mapping[str, object]: Selected result, judgment and terminal outcome.

        Raises:
            ValueError: If results and assessments are not linked.
        """
        text: str | None = None
        assessment: bool | None = None
        error: str | None = None
        for event in events:
            match event:
                case {"event": "result", "text": str() as value} if assessment is not None or (text is None and error is None):
                    text = value
                    assessment = None
                case {"event": "judgment", "text": str() as value, "assessment": bool() as verdict} if text is not None and assessment is None and error is None and value == text:
                    assessment = verdict
                case {"event": "assessment_error", "text": str() as value, "error": str() as detail} if text is not None and assessment is None and error is None and value == text:
                    error = detail
                case {"event": "end"}:
                    return self.finish(
                        event, text, assessment, error
                    )
                case {"event": "result" | "judgment" | "assessment_error"}:
                    raise ValueError("Invalid run journal event")
                case _:
                    continue
        raise ValueError("Run journal has no outcome")

    @override
    def finish(self, event: dict[str, object], text: str | None, assessment: bool | None, error: str | None) -> Mapping[str, object]:
        """Link the terminal event to the selected result.

        Args:
            event (dict[str, object]): Recorded end event.
            text (str | None): Selected text.
            assessment (bool | None): Linked verdict.
            error (str | None): Assessment error.

        Returns:
            Mapping[str, object]: Recovered outcome.

        Raises:
            ValueError: If the terminal event is inconsistent.
        """
        match event:
            case {"status": str() as status, "reason": str() as reason}:
                self.validate(
                    status, reason=reason, text=text, assessment=assessment, error=error
                )
            case {"status": str() as status} if REASON not in event:
                self.validate(
                    status, reason=status, text=text, assessment=assessment, error=error
                )
                reason = status
            case _:
                raise ValueError(f"Invalid run journal event in {self.path}")
        return {
            "status": status,
            "reason": reason,
            "text": text,
            "assessment": assessment,
            "assessment_error": error
        }

    @override
    def validate(self, status: str, *, reason: str, text: str | None, assessment: bool | None, error: str | None) -> None:
        """Require a terminal status to agree with its recorded result.

        Args:
            status (str): Saved terminal status.
            reason (str): Saved terminal reason.
            text (str | None): Selected text, if present.
            assessment (bool | None): Linked verdict, if present.
            error (str | None): Assessment failure, if present.

        Raises:
            ValueError: If the outcome disagrees with its assessment.
        """
        if not (
            (status == COMPLETED and text is not None and assessment is not None)
            or (status == SUCCESS and text is not None and assessment is True)
            or (status == ROUND_LIMIT and text is not None and assessment is False)
            or (status == NO_RESULT and text is None and assessment is None)
            or (status == FAILURE and error is not None and assessment is None)
            or (status == FAILURE and reason == EXECUTION_ERROR and error is None)
        ):
            raise ValueError(f"Invalid run journal event in {self.path}")

    @override
    def metrics(self, events: Sequence[dict[str, object]]) -> Mapping[str, object]:
        """Count role calls and conversation decisions.

        Args:
            events (Sequence[dict[str, object]]): Saved exchanges and decisions.

        Returns:
            Mapping[str, object]: Role calls, refusals and rollbacks.
        """
        refusals = 0
        rollbacks = 0
        for event in events:
            match event:
                case {"event": "step", "accepted": False}:
                    refusals += 1
                case {"event": "decision", "discarded": list()}:
                    if event["discarded"] != []:
                        rollbacks += 1
                case _:
                    continue
        return {
            "calls": self.calls(events),
            "refusals": refusals,
            "rollbacks": rollbacks
        }

    @override
    def duration(self, events: Sequence[dict[str, object]]) -> float | None:
        """Measure elapsed wall time from saved boundary timestamps.

        Args:
            events (Sequence[dict[str, object]]): Saved run events.

        Returns:
            float | None: Seconds, or absent for historical untimed journals.

        Raises:
            ValueError: If the saved end precedes the start.
        """
        if not events:
            return None
        if events[0].get("event") == SERIES:
            events = events[1:]
        match events:
            case [{"event": "start", "at": int() | float() as start}, *_, {"event": "end", "at": int() | float() as end}]:
                if end < start:
                    raise ValueError(f"Invalid run journal duration in {self.path}")
                return end - start
            case _:
                return None

    @override
    def calls(self, events: Iterable[dict[str, object]]) -> dict[str, int]:
        """Count calls while checking the surrounding event sequence.

        Args:
            events (Iterable[dict[str, object]]): Events from the finished journal.

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
                case ({"event": "series", "summary": str()}
                      | {"event": "start", "base": str()}) if not started:
                    started = event["event"] == START
                case {"event": "call", "role": str() as role, "prompt": str(), "response": str() | None} if started and role in counts:
                    counts[role] += 1
                case ({"event": "step", "prompt": str(), "response": str(), "accepted": bool()}
                       | {"event": "result", "text": str()}
                       | {"event": "judgment", "text": str(), "assessment": bool()}
                       | {"event": "assessment_error", "text": str(), "error": str()}
                       | {"event": "execution_error", "error": str()}
                       | {"event": "decision", "action": str(), "discarded": list(), "kept": int()}) if started:
                    pass
                case {"event": "end", "status": "completed" | "success" | "round_limit" | "no_result" | "failure"} if started:
                    finished = True
                case _:
                    raise ValueError(f"Invalid run journal event in {self.path}")
        return counts

    @override
    def events(self) -> Iterator[dict[str, object]]:
        """Yield saved events in their original order.

        Yields:
            dict[str, object]: An event decoded from the journal.

        Raises:
            ValueError: If an event has an unsupported journal schema.
        """
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                event: dict[str, object] = json.loads(line)
                if event.get("schema_id") != SCHEMA_ID or event.get("schema_version") != 1:
                    raise ValueError(f"Unsupported journal schema in {self.path}: {event.get('schema_id')!r} version {event.get('schema_version')!r}")
                yield event
