"""Independent, fail-fast diagnostics for one execution.

The journal orders, timestamps, snapshots, formats and stores records. Control
requests enqueue without storage I/O; the worker flushes before its next operation.
A storage failure poisons the journal permanently.
"""

import json
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, replace
from pathlib import Path
from typing import Unpack, override

from .journal_calls import (
    ASSESSMENT, EXCHANGE, STOP_REQUESTED, CallInputs, call_data, failure_details,
)
from .journal_formatting import event_details
from .journal_lifecycle import JournalLifecycle
from .journal_output import render_report
from .journal_ports import Journal
from .journal_values import operation_value
from .operations import EvaluationResult, Exchange, WorkerOutcome
from .records import RunEvent
from .status import UserState

__all__ = ["RunEvent", "RunJournal"]


class RunJournal(JournalLifecycle, Journal):
    @override
    def events(self, after: int = 0) -> tuple[RunEvent, ...]:
        with self._state.lock:
            return tuple(self._output.events[after:])

    @override
    def bind(self, run_id: str) -> None:
        """Bind admission without I/O before worker and control can race.

        Args:
            run_id (str): Admitted run identity.

        Raises:
            RuntimeError: The journal already belongs to a run.
        """
        self.check()
        if self._state.run_id is not None:
            raise RuntimeError("Journal already belongs to a run")
        self._state.run_id = run_id
        with self._state.lock:
            self._state.status = replace(self._state.status, run_id=run_id, state=UserState.RUNNING)
        self.enqueue("run_started")

    @override
    def begin(self, run_id: str, directory: Path | None = None) -> None:
        self.check()
        if self._output.begun:
            raise RuntimeError("Journal already started")
        if self._state.run_id is None:
            self.bind(run_id)
        if run_id != self._state.run_id:
            raise ValueError("Journal belongs to another run")
        self._output.begun = True
        with self._guard():
            self._output.open(directory)
            self.flush()

    @override
    def configure(self) -> None:
        self.check()
        with self._guard():
            self.record("run_configured", settings=self._capture_initial_settings())

    @override
    def enqueue(self, kind: str, **data: object) -> None:
        """Record control-side evidence without worker-owned storage I/O.

        Args:
            kind (str): Event kind.
            **data (object): Explicit event evidence.
        """
        self.check()
        with self._guard():
            snapshot = json.dumps(self.redactor.snapshot(operation_value(data)), ensure_ascii=False)
            with self._state.lock:
                self._output.append(self._state.run_id or "", kind, snapshot, self.redactor)
                if kind == STOP_REQUESTED:
                    self._state.status = replace(self._state.status, state=UserState.STOPPING)

    @override
    def record(self, kind: str, **data: object) -> None:
        self.enqueue(kind, **data)
        self.flush()

    def _write_event(self, event: RunEvent) -> None:
        self._output.write(event)

    @override
    def flush(self) -> None:
        self.check()
        with self._guard():
            pending = self.events(self._output.flushed)
            for event in pending:
                self._write_event(event)
                self._output.flushed += 1

    def _start_call(self, operation: str, data: dict[str, object]) -> None:
        if operation == EXCHANGE:
            self.record("call_started", **data)
        elif operation == ASSESSMENT:
            self._state.increment("evaluator_calls")
            self.record("evaluation_started", **data)
        else:
            self.record("operation_started", operation=operation, **data)

    @override
    def call[T](
        self, operation: str, invoke: Callable[[], T], *,
        project: Callable[[T], object] | None = None, **inputs: Unpack[CallInputs],
    ) -> T:
        """Wrap admitted work using only explicit inputs and its return.

        Args:
            operation (str): Operation name; exchange and assessment have dedicated events.
            invoke (Callable[[], T]): Admitted operation to execute once.
            project (Callable[[T], object] | None, default=None): Observable outcome selection.
            **inputs (Unpack[CallInputs]): Correlation and evidence supplied by the caller.

        Returns:
            T: The original operation's outcome.

        Raises:
            Exception: The original operation error after recording available evidence.
        """
        data = call_data(operation, inputs)
        self._start_call(operation, data)
        try:
            outcome = invoke()
        except Exception as exc:
            self.check()  # Never retry a poisoned writer, even after library translation.
            if operation in {EXCHANGE, ASSESSMENT}:
                kind, details = failure_details(operation, exc)
                self.record(kind, **data, **details)
            raise
        self.check()
        self.observe(
            project(outcome) if project is not None else outcome,
            result_id=inputs.get("result_id"), evaluation_id=inputs.get("evaluation_id"),
        )
        return outcome

    @override
    def observe(
        self, outcome: object, *, result_id: str | None = None,
        evaluation_id: str | None = None,
    ) -> None:
        """Format domain outcomes without repeating decisions or native checks.

        Args:
            outcome (object): Returned domain operation value.
            result_id (str | None, default=None): Correlated presented result.
            evaluation_id (str | None, default=None): Correlated assessment identity.
        """
        if (details := event_details(outcome)) is None:
            return
        if isinstance(outcome, Exchange):
            details.data.update({key: value for key, value in {
                "result_id": result_id, "evaluation_id": evaluation_id,
            }.items() if value is not None})
        elif isinstance(outcome, EvaluationResult):
            details.data["evaluation_id"] = evaluation_id
        self._state.observe(outcome)
        self.record(details.kind, **details.data)

    def _finish_records(self, outcome: WorkerOutcome) -> None:
        self.check()
        if outcome.error is not None:
            self.record(
                "run_error", error=f"{type(outcome.error).__name__}: {outcome.error}",
                diagnostic=operation_value(outcome.error),
            )
        if outcome.released is not None:
            self.record("resources_released", **asdict(outcome.released))
        self.record("run_finished", state=outcome.status.state.value, reason=outcome.status.reason)

    def _finish_report(
        self, outcome: WorkerOutcome, report: tuple[Mapping[str, object], Path | None] | None,
    ) -> str | None:
        self._finish_records(outcome)
        if report is None:
            return None
        data, path = report
        with self._guard():
            return render_report({
                **data, "events": [json.loads(event.dump()) for event in self.events()],
            }, path, self.redactor)

    def _close(self, outcome: WorkerOutcome) -> None:
        # Close even a poisoned writer, preserving the first failure over close failures.
        with suppress(Exception), self._guard():
            self._output.close()
        self._state.finish(outcome, self.redactor)

    @override
    def finish(
        self, outcome: WorkerOutcome, *,
        report: tuple[Mapping[str, object], Path | None] | None = None,
    ) -> str | None:
        """Finish after resource release, closing the journal even on failure.

        Args:
            outcome (WorkerOutcome): Actual completed worker outcome.
            report (tuple[Mapping[str, object], Path | None] | None, default=None):
                Optional report evidence/path.

        Returns:
            str | None: Rendered report, when requested.

        Raises:
            RuntimeError: The journal is already closed.
            error: The first journal failure, including a close failure.
        """
        if self.finished:
            raise RuntimeError("Journal is closed")
        try:
            rendered = self._finish_report(outcome, report)
        finally:
            self._close(outcome)
        if (error := self.error) is not None:
            raise error
        return rendered
