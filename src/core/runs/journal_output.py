"""Ordered event snapshots and the worker-owned journal stream."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO, override

from core.primitives.diagnostics import DiagnosticRedactor

from .journal_values import operation_value
from .journal_support import JournalOutputPort
from .records import RunEvent


@dataclass
class JournalOutput(JournalOutputPort):
    events: list[RunEvent] = field(default_factory=list[RunEvent])
    flushed: int = 0
    begun: bool = False
    _stream: TextIO | None = None

    @override
    def open(self, directory: Path | None) -> None:
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=False)
            self._stream = (directory / "events.jsonl").open("w", encoding="utf-8")

    @override
    def append(
        self, run_id: str, kind: str, snapshot: str, redactor: DiagnosticRedactor
    ) -> None:
        self.events.append(RunEvent(
            run_id, len(self.events) + 1, kind, datetime.now(UTC).isoformat(), snapshot, redactor
        ))

    @override
    def write(self, event: RunEvent) -> None:
        if self._stream is not None:
            _ = self._stream.write(event.dump() + "\n")
            self._stream.flush()

    @override
    def close(self) -> None:
        try:
            self._close_stream()
        finally:
            self._stream = None

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.close()


def render_report(data: object, path: Path | None, redactor: DiagnosticRedactor) -> str:
    """Format and optionally store a detached diagnostic report.

    Args:
        data (object): Explicit report evidence.
        path (Path | None): Optional report destination.
        redactor (DiagnosticRedactor): Run diagnostic policy.

    Returns:
        str: Rendered JSON report including a final newline.
    """
    rendered = json.dumps(
        redactor.snapshot(operation_value(data)), ensure_ascii=False, indent=2
    ) + "\n"
    if path is not None:
        _ = path.write_text(rendered, encoding="utf-8")
    return rendered
