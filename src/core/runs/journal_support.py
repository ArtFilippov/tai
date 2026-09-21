"""Ports for the synchronized projection and ordered output used by a journal.

The shared state lock makes event admission and its status transition atomic.
Output buffer access is protected by that lock; only the worker writes or closes
the output sink. These contracts keep synchronization and storage implementations
out of observation and retained-evidence consumers.
"""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from core.primitives.diagnostics import DiagnosticRedactor

from .operations import WorkerOutcome
from .records import RunEvent
from .status import UserStatus


class JournalStatePort(Protocol):
    status: UserStatus
    lock: AbstractContextManager[bool]
    run_id: str | None
    error: Exception | None

    def finished(self) -> bool: ...

    def wait(self, timeout: float | None) -> bool: ...

    def poison(self, error: Exception, redactor: DiagnosticRedactor) -> None: ...

    def increment(self, counter: str) -> None: ...

    def observe(self, outcome: object) -> None: ...

    def finish(self, outcome: WorkerOutcome, redactor: DiagnosticRedactor) -> None: ...


class JournalOutputPort(Protocol):
    events: list[RunEvent]
    flushed: int
    begun: bool

    def open(self, directory: Path | None) -> None: ...

    def append(
        self, run_id: str, kind: str, snapshot: str, redactor: DiagnosticRedactor,
    ) -> None: ...

    def write(self, event: RunEvent) -> None: ...

    def close(self) -> None: ...
