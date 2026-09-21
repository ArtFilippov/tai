"""Dependency contracts for execution, observation, clocks, and persistence."""

from abc import abstractmethod
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, Unpack, override

from core.primitives.diagnostics import DiagnosticObject, DiagnosticRedactor
from core.primitives.messages import Message
from core.runs import RunSnapshot, UserStatus
from core.runs.operations import Context, WorkerOutcome


class EventView(Protocol):
    @property
    def sequence(self) -> int: ...

    @property
    def kind(self) -> str: ...

    @property
    def data(self) -> DiagnosticObject: ...


class JournalView(Protocol):
    @property
    def error(self) -> Exception | None: ...

    def status(self) -> UserStatus: ...

    def events(self, after: int = 0) -> tuple[EventView, ...]: ...


class RunControl(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...

    def status(self) -> UserStatus: ...

    def snapshot(self) -> RunSnapshot: ...


class ObservationInputs(TypedDict, total=False):
    context: Context | None
    call_id: str | None
    message: Message | None
    result_id: str | None
    base: Message | None
    result: Message | None
    evaluation_id: str | None
    checkpoint_id: str | None
    settings: DiagnosticObject | None


class ReadinessJournal(JournalView, Protocol):
    @property
    def redactor(self) -> DiagnosticRedactor: ...

    def begin(self, run_id: str, directory: Path | None = None) -> None: ...

    def check(self) -> None: ...

    def call[T](
        self, operation: str, invoke: Callable[[], T], *,
        project: Callable[[T], object] | None = None, **inputs: Unpack[ObservationInputs],
    ) -> T: ...

    @abstractmethod
    def finish(self, outcome: WorkerOutcome, *, report: tuple[Mapping[str, object], Path | None] | None = None) -> str | None: ...


class StopRequests(Protocol):
    @property
    def interrupted(self) -> bool: ...

    def request(self) -> None: ...

    def seal(self) -> None: ...


@dataclass(frozen=True)
class Clock:
    now: Callable[[], str]
    elapsed: Callable[[], float]


@dataclass(frozen=True)
class ReadinessServices:
    journal: ReadinessJournal
    observe: Callable[[Context, Message, Message], AbstractContextManager[str]]
    clock: Clock
    version: str


@dataclass(frozen=True)
class RunHandle(RunControl):
    _user: RunControl
    journal: Callable[[], JournalView]
    save: Callable[[], None]

    @override
    def start(self) -> None:
        self._user.start()

    @override
    def stop(self) -> None:
        self._user.stop()

    @override
    def wait(self, timeout: float | None = None) -> bool:
        return self._user.wait(timeout)

    @override
    def status(self) -> UserStatus:
        return self._user.status()

    @override
    def snapshot(self) -> RunSnapshot:
        return self._user.snapshot()
