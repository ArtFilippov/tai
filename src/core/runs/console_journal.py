"""Direct console observation, retaining RunJournal's ordinary file evidence."""

import sys
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Unpack, override

from core.primitives.diagnostics import DiagnosticRedactor

from .console_ports import ConsoleRow, DisplayExchange, RowChange
from .journal import RunJournal
from .journal_calls import EXCHANGE, CallInputs
from .journal_configuration import JournalOptions
from .journal_delegate import JournalDelegate
from .operations import Exchange, WorkerOutcome


class ConsoleRunJournal(JournalDelegate):
    def __init__(
        self, row: ConsoleRow, *, redactor: DiagnosticRedactor,
        **options: Unpack[JournalOptions],
    ) -> None:
        super().__init__(RunJournal(redactor=redactor, **options))
        self._row = row
        self._methods: list[str] = []

    def _update(self, change: RowChange) -> None:
        try:
            self._row.update(change)
        except Exception as exc:
            self._console_error = self.error or exc
            raise

    @override
    def bind(self, run_id: str) -> None:
        self.check()
        self._journal.bind(run_id)
        self._update(RowChange(state="RUNNING", run_id=run_id))

    @override
    def begin(self, run_id: str, directory: Path | None = None) -> None:
        self.check()
        self._journal.begin(run_id, directory)

    @override
    def configure(self) -> None:
        self.check()
        self._journal.configure()

    def _enter(self, operation: str, inputs: CallInputs) -> None:
        context = inputs.get("context")
        if operation == EXCHANGE and context is not None:
            operation += f"[{context.role}]"
        self._methods.append(operation)
        self._update(RowChange(method=operation))

    def _leave(self) -> None:
        _ = self._methods.pop()
        self._update(RowChange(method=self._methods[-1] if self._methods else ""))

    @contextmanager
    def _method(self, operation: str, inputs: CallInputs) -> Generator[None]:
        self._enter(operation, inputs)
        try:
            yield
        finally:
            if sys.exception() is not None:
                with suppress(Exception):
                    self._leave()
            else:
                self._leave()

    @override
    def call[T](
        self, operation: str, invoke: Callable[[], T], *,
        project: Callable[[T], object] | None = None, **inputs: Unpack[CallInputs],
    ) -> T:
        self.check()
        with self._method(operation, inputs):
            result = self._journal.call(operation, invoke, project=project, **inputs)
            self._show(result)
            return result

    @override
    def observe(self, outcome: object, *, result_id: str | None = None, evaluation_id: str | None = None) -> None:
        self.check()
        self._journal.observe(outcome, result_id=result_id, evaluation_id=evaluation_id)
        self._show(outcome)

    def _show(self, outcome: object) -> None:
        if isinstance(outcome, Exchange):
            self._update(RowChange(exchange=DisplayExchange(
                outcome.context.role, outcome.call_id,
                self.redactor.text(outcome.request.text), self.redactor.text(outcome.response.text),
            )))

    def _terminal(self, outcome: WorkerOutcome) -> None:
        self._methods.clear()
        failed = self.error is not None or outcome.error is not None
        self._update(RowChange(state="FAILED" if failed else outcome.status.state.value, method=""))

    @override
    def finish(self, outcome: WorkerOutcome, *, report: tuple[Mapping[str, object], Path | None] | None = None) -> str | None:
        try:
            return self._journal.finish(outcome, report=report)
        finally:
            if sys.exception() is not None:
                with suppress(Exception):
                    self._terminal(outcome)
            else:
                self._terminal(outcome)
