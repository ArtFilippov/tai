"""Journal contracts for observation, retained evidence, and worker finalization.

The lifecycle view supports observation-independent reads and failure propagation.
The complete journal adds ordered evidence and admitted-operation observation;
neither contract exposes a storage handle or a concrete journal implementation.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, Unpack

from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.diagnostics.json_values import JSONValue

from .journal_calls import CallInputs
from .operations import WorkerOutcome
from .records import RunEvent
from .status import UserStatus


class JournalLifecyclePort(Protocol):
    redactor: DiagnosticRedactor

    @property
    def error(self) -> Exception | None: ...

    @property
    def finished(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...

    def status(self) -> UserStatus: ...

    def check(self) -> None: ...

    def settings(self) -> JSONValue: ...

    def final_settings(self) -> JSONValue: ...

    def capture_final_settings(self) -> None: ...


class Journal(JournalLifecyclePort, Protocol):
    def events(self, after: int = 0) -> tuple[RunEvent, ...]: ...

    def bind(self, run_id: str) -> None: ...

    def begin(self, run_id: str, directory: Path | None = None) -> None: ...

    def configure(self) -> None: ...

    def enqueue(self, kind: str, **data: object) -> None: ...

    def record(self, kind: str, **data: object) -> None: ...

    def flush(self) -> None: ...

    def call[T](
        self, operation: str, invoke: Callable[[], T], *,
        project: Callable[[T], object] | None = None, **inputs: Unpack[CallInputs],
    ) -> T: ...

    def observe(
        self, outcome: object, *, result_id: str | None = None,
        evaluation_id: str | None = None,
    ) -> None: ...

    def finish(
        self, outcome: WorkerOutcome, *,
        report: tuple[Mapping[str, object], Path | None] | None = None,
    ) -> str | None: ...
