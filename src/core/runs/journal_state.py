"""Synchronized lifecycle and immutable status updates for a journal."""

from dataclasses import dataclass, field, replace
from contextlib import AbstractContextManager
from threading import Event, Lock
from typing import override

from core.primitives.diagnostics import DiagnosticRedactor

from .operations import EvaluationResult, Exchange, ResultChange, WorkerOutcome
from .journal_support import JournalStatePort
from .status import UserState, UserStatus


@dataclass
class JournalState(JournalStatePort):
    status: UserStatus
    lock: AbstractContextManager[bool] = field(default_factory=Lock)
    done: Event = field(default_factory=Event)
    run_id: str | None = None
    error: Exception | None = None

    @override
    def finished(self) -> bool:
        return self.done.is_set()

    @override
    def wait(self, timeout: float | None) -> bool:
        return self.done.wait(timeout)

    @override
    def poison(self, error: Exception, redactor: DiagnosticRedactor) -> None:
        with self.lock:
            if self.error is None:
                self.error = error
                self.status = replace(
                    self.status, state=UserState.FAILED,
                    reason=redactor.text(f"{type(error).__name__}: {error}"),
                )

    @override
    def increment(self, counter: str) -> None:
        with self.lock:
            progress = self.status.progress
            self.status = replace(self.status, progress=replace(
                progress, **{counter: getattr(progress, counter) + 1}
            ))

    @override
    def observe(self, outcome: object) -> None:
        if isinstance(outcome, Exchange):
            self._exchange(outcome)
        elif isinstance(outcome, ResultChange):
            with self.lock:
                self.status = replace(
                    self.status, result_id=outcome.current.id if outcome.current else None,
                    evaluation=outcome.evaluation,
                )
        elif isinstance(outcome, EvaluationResult):
            with self.lock:
                self.status = replace(self.status, evaluation=outcome.evaluation)

    def _exchange(self, outcome: Exchange) -> None:
        counter = {"Target": "target_calls", "Attacker": "attacker_calls"}.get(
            outcome.context.role
        )
        if counter is not None:
            self.increment(counter)

    @override
    def finish(self, outcome: WorkerOutcome, redactor: DiagnosticRedactor) -> None:
        with self.lock:
            status = outcome.status
            if self.error is not None:
                primary = outcome.error or self.error
                status = replace(
                    status, state=UserState.FAILED, reason=f"{type(primary).__name__}: {primary}"
                )
            self.status = replace(
                status, reason=redactor.text(status.reason) if status.reason else status.reason
            )
        self.done.set()
