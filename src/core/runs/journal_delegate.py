"""A journal decorator forwards lifecycle/evidence through the public contract."""

from typing import override

from core.primitives.diagnostics import DiagnosticRedactor

from .journal_ports import Journal
from .status import UserStatus
from .records import RunEvent


class JournalDelegate(Journal):
    def __init__(self, journal: Journal) -> None:
        super().__init__()
        self._journal = journal
        self._console_error: Exception | None = None
        self.redactor: DiagnosticRedactor = journal.redactor

    @property
    @override
    def error(self) -> Exception | None:
        return self._console_error or self._journal.error

    @property
    @override
    def finished(self) -> bool:
        return self._journal.finished

    @override
    def wait(self, timeout: float | None = None) -> bool:
        return self._journal.wait(timeout)

    @override
    def status(self) -> UserStatus:
        return self._journal.status()

    @override
    def check(self) -> None:
        if (error := self.error) is not None:
            raise error
        self._journal.check()

    @override
    def settings(self) -> object:
        return self._journal.settings()

    @override
    def final_settings(self) -> object:
        return self._journal.final_settings()

    @override
    def capture_final_settings(self) -> None:
        self.check()
        self._journal.capture_final_settings()

    @override
    def events(self, after: int = 0) -> tuple[RunEvent, ...]:
        return self._journal.events(after)

    @override
    def enqueue(self, kind: str, **data: object) -> None:
        self.check()
        self._journal.enqueue(kind, **data)

    @override
    def record(self, kind: str, **data: object) -> None:
        self.check()
        self._journal.record(kind, **data)

    @override
    def flush(self) -> None:
        self.check()
        self._journal.flush()
