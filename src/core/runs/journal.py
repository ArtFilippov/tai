"""Independent, fail-fast diagnostics for one execution.

Only this object orders, timestamps, snapshots, formats and stores records.
Control requests enqueue without storage I/O; the worker flushes them before its
next operation. A storage failure poisons the journal permanently.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock
from typing import Any, TextIO

from core.chats import ChatCancelled
from core.primitives.diagnostics import DiagnosticObject, DiagnosticRedactor
from core.primitives.messages import Message

from .operations import (
    BranchDecision,
    Checkpoint,
    Context,
    ConversationChange,
    ConversationPoint,
    EvaluationResult,
    Exchange,
    LocalDecision,
    ResultChange,
    Synchronization,
    TargetExchange,
    WorkerOutcome,
)
from .status import Progress, UserState, UserStatus


def _value(value: Any) -> Any:
    """Serialize explicit operation values; never inspect a checkpoint token."""
    if isinstance(value, Message):
        return json.loads(value.dump())
    if isinstance(value, Checkpoint):
        return {"context": _value(value.context), "id": value.id}
    if isinstance(value, Exception):
        return {
            "type": type(value).__name__,
            "message": str(value),
            "diagnostic": getattr(value, "diagnostic", None),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _value(item) for key, item in value.items()}
    return value


def _context(context: Context) -> dict[str, Any]:
    return {
        "context": _value(context),
        "role": context.role,
        "conversation_id": context.conversation_id,
        "attempt_id": context.cause.attempt_id,
        "cause_id": context.cause.id,
    }


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence: int
    kind: str
    timestamp: str
    _data_json: str = field(repr=False)
    _redactor: DiagnosticRedactor = field(
        default_factory=DiagnosticRedactor, repr=False, compare=False
    )

    @property
    def data(self) -> DiagnosticObject:
        """A detached diagnostic snapshot, never a source for a live Message."""
        return json.loads(self._data_json)

    def dump(self) -> str:
        return json.dumps(
            self._redactor.snapshot(
                {
                    "run_id": self.run_id,
                    "sequence": self.sequence,
                    "kind": self.kind,
                    "timestamp": self.timestamp,
                    "data": self.data,
                }
            ),
            ensure_ascii=False,
        )


class RunJournal:
    def __init__(
        self,
        *,
        redactor: DiagnosticRedactor | None = None,
        max_calls_per_conv: int | None = None,
        settings: Callable[[], dict[str, Any]] = dict,
        final_settings: Callable[[], dict[str, Any]] = dict,
    ) -> None:
        self.redactor = redactor or DiagnosticRedactor()
        self._settings_source = settings
        self._final_settings_source = final_settings
        self._settings: Any = None
        self._final_settings: Any = None
        self._events: list[RunEvent] = []
        self._flushed = 0
        self._lock = Lock()
        self._stream: TextIO | None = None
        self._begun = False
        self._run_id: str | None = None
        self._error: Exception | None = None
        self._done = Event()
        self._status = UserStatus(
            progress=Progress(max_calls_per_conv=max_calls_per_conv)
        )

    @property
    def error(self) -> Exception | None:
        """Original diagnostic/storage failure, available even if storage failed."""
        with self._lock:
            return self._error

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for actual work and journal closure, independently of User.wait."""
        return self._done.wait(timeout)

    def events(self, after: int = 0) -> tuple[RunEvent, ...]:
        with self._lock:
            return tuple(self._events[after:])

    def _render_report(self, data: Any, path: Path | None = None) -> str:
        """Format and optionally store a detached diagnostic report."""
        if self.error is not None:
            raise self.error
        try:
            rendered = (
                json.dumps(
                    self.redactor.snapshot(_value(data)), ensure_ascii=False, indent=2
                )
                + "\n"
            )
            if path is not None:
                path.write_text(rendered, encoding="utf-8")
            return rendered
        except Exception as exc:
            self._poison(exc)
            raise

    def status(self) -> UserStatus:
        with self._lock:
            return self._status

    def _poison(self, error: Exception) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
                self._status = replace(
                    self._status,
                    state=UserState.FAILED,
                    reason=self.redactor.text(f"{type(error).__name__}: {error}"),
                )

    def check(self) -> None:
        error = self.error
        if error is not None:
            raise error
        if self.finished:
            raise RuntimeError("Journal is closed")

    def bind(self, run_id: str) -> None:
        """Bind admission without I/O, before the worker and control can race."""
        self.check()
        if self._run_id is not None:
            raise RuntimeError("Journal already belongs to a run")
        self._run_id = run_id
        with self._lock:
            self._status = replace(self._status, run_id=run_id, state=UserState.RUNNING)
        self.enqueue("run_started")

    def begin(self, run_id: str, directory: Path | None = None) -> None:
        self.check()
        if self._begun:
            raise RuntimeError("Journal already started")
        if self._run_id is None:
            self.bind(run_id)
        if run_id != self._run_id:
            raise ValueError("Journal belongs to another run")
        self._begun = True
        try:
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=False)
                self._stream = (directory / "events.jsonl").open("w", encoding="utf-8")
            self.flush()
        except Exception as exc:
            self._poison(exc)
            raise

    def configure(self) -> None:
        self.check()
        try:
            self._settings = self.redactor.snapshot(_value(self._settings_source()))
            self.record("run_configured", settings=self._settings)
        except Exception as exc:
            self._poison(exc)
            raise

    def enqueue(self, kind: str, **data: Any) -> None:
        """Nonblocking control-side recording; storage is worker-owned."""
        self.check()
        try:
            snapshot = json.dumps(
                self.redactor.snapshot(_value(data)), ensure_ascii=False
            )
            with self._lock:
                self._events.append(
                    RunEvent(
                        self._run_id or "",
                        len(self._events) + 1,
                        kind,
                        datetime.now(UTC).isoformat(),
                        snapshot,
                        self.redactor,
                    )
                )
                if kind == "stop_requested":
                    self._status = replace(self._status, state=UserState.STOPPING)
        except Exception as exc:
            self._poison(exc)
            raise

    def record(self, kind: str, **data: Any) -> None:
        self.enqueue(kind, **data)
        self.flush()

    def _write_event(self, event: RunEvent) -> None:
        if self._stream is not None:
            self._stream.write(event.dump() + "\n")
            self._stream.flush()

    def flush(self) -> None:
        self.check()
        try:
            with self._lock:
                pending = tuple(self._events[self._flushed :])
            for event in pending:
                self._write_event(event)
                self._flushed += 1
        except Exception as exc:
            self._poison(exc)
            raise

    def call(
        self,
        operation: str,
        invoke: Callable[[], Any],
        *,
        context: Context | None = None,
        call_id: str | None = None,
        message: Message | None = None,
        result_id: str | None = None,
        checkpoint_id: str | None = None,
        base: Message | None = None,
        result: Message | None = None,
        evaluation_id: str | None = None,
        settings: DiagnosticObject | None = None,
        project: Callable[[Any], Any] = lambda value: value,
    ) -> Any:
        """Wrap an admitted operation using only explicit inputs and its return."""
        data = _context(context) if context is not None else {}
        if result_id is not None:
            data["result_id"] = result_id
        if checkpoint_id is not None:
            data["checkpoint_id"] = checkpoint_id
        if base is not None:
            data.update(base_id=base.id, base=_value(base))
        if result is not None:
            data["result"] = _value(result)
        if evaluation_id is not None:
            data["evaluation_id"] = evaluation_id
        if settings is not None:
            data["settings"] = settings
        if operation == "exchange":
            data.update(call_id=call_id, input=_value(message))
            self.record("call_started", **data)
        elif operation == "assessment":
            with self._lock:
                self._status = replace(
                    self._status,
                    progress=replace(
                        self._status.progress,
                        evaluator_calls=self._status.progress.evaluator_calls + 1,
                    ),
                )
            self.record("evaluation_started", **data)
        else:
            self.record("operation_started", operation=operation, **data)
        try:
            outcome = invoke()
        except Exception as exc:
            # Never retry a poisoned writer, even if garak translated its error.
            self.check()
            if operation in ("exchange", "assessment"):
                prefix = "call" if operation == "exchange" else "evaluation"
                cancelled = isinstance(exc, ChatCancelled)
                details = {"error": f"{type(exc).__name__}: {exc}"}
                diagnostic = getattr(exc, "diagnostic", None)
                if diagnostic is not None:
                    details["diagnostic"] = diagnostic
                if isinstance(exc, ChatCancelled) and operation == "exchange":
                    details["response"] = _value(exc.response)
                self.record(
                    prefix + ("_cancelled" if cancelled else "_failed"),
                    **data,
                    **details,
                )
            raise
        self.check()
        self.observe(project(outcome), result_id=result_id, evaluation_id=evaluation_id)
        return outcome

    def observe(
        self,
        outcome: Any,
        *,
        result_id: str | None = None,
        evaluation_id: str | None = None,
    ) -> None:
        """Format domain outcomes. No decisions or native checks are repeated."""
        if isinstance(outcome, Exchange):
            data = dict(
                _context(outcome.context),
                call_id=outcome.call_id,
                request_id=outcome.request.id,
                response=_value(outcome.response),
            )
            if result_id is not None:
                data["result_id"] = result_id
            if evaluation_id is not None:
                data["evaluation_id"] = evaluation_id
            counter = {
                "Target": "target_calls",
                "Attacker": "attacker_calls",
                "Evaluator": "evaluator_calls",
            }[outcome.context.role]
            if outcome.context.role != "Evaluator":
                with self._lock:
                    progress = self._status.progress
                    self._status = replace(
                        self._status,
                        progress=replace(
                            progress, **{counter: getattr(progress, counter) + 1}
                        ),
                    )
            self.record("call_completed", **data)
        elif isinstance(outcome, Checkpoint):
            self.record(
                "checkpoint_captured",
                **_context(outcome.context),
                checkpoint_id=outcome.id,
            )
        elif isinstance(outcome, ConversationPoint):
            self.record(
                "checkpoint_created",
                **_context(outcome.checkpoint.context),
                checkpoint_id=outcome.checkpoint.id,
                request_id=outcome.request.id if outcome.request else None,
                response_id=outcome.response.id if outcome.response else None,
                result_id=outcome.result.id if outcome.result else None,
            )
        elif isinstance(outcome, ConversationChange):
            self.record(
                "history_" + outcome.kind,
                **_context(outcome.context),
                previous_conversation_id=outcome.previous_conversation_id,
                checkpoint_id=outcome.checkpoint.id if outcome.checkpoint else None,
            )
        elif isinstance(outcome, Synchronization):
            self.record(
                "conversation_synchronized",
                outcome=_value(outcome),
                **_context(outcome.conversation.context),
                from_checkpoint_id=outcome.before.checkpoint.id,
                checkpoint_id=outcome.after.checkpoint.id,
            )
        elif isinstance(outcome, TargetExchange):
            self.record(
                "target_exchange",
                **_context(outcome.exchange.context),
                call_id=outcome.exchange.call_id,
                checkpoint_id=outcome.point.checkpoint.id,
                request_id=outcome.exchange.request.id,
                response_id=outcome.exchange.response.id,
            )
        elif isinstance(outcome, LocalDecision):
            self.record(
                "local_decision",
                **_context(outcome.context),
                decision=outcome.kind,
                value=outcome.value,
                request_id=outcome.request.id,
                response_id=outcome.response.id,
                checkpoint_id=outcome.checkpoint_id,
                judgment=_value(outcome.judgment),
                judgment_response_id=outcome.judgment.response.id
                if outcome.judgment
                else None,
            )
        elif isinstance(outcome, BranchDecision):
            self.record(
                "branch_selected",
                **_context(outcome.context),
                branch_id=outcome.id,
                branch=outcome.branch,
                point=_value(outcome.point),
                next_attempt_id=outcome.next_attempt_id,
                next_request=_value(outcome.next_request),
                response_id=outcome.point.response.id
                if outcome.point.response
                else None,
                checkpoint_id=outcome.point.checkpoint.id,
                levels_completed=outcome.levels_completed,
                next_levels_completed=outcome.next_levels_completed,
                checks=_value(outcome.checks),
            )
        elif isinstance(outcome, ResultChange):
            with self._lock:
                self._status = replace(
                    self._status,
                    result_id=outcome.current.id if outcome.current else None,
                    evaluation=outcome.evaluation,
                )
            self.record(
                "result_"
                + (
                    "withdrawn"
                    if outcome.kind == "withdrawn"
                    else "unchanged"
                    if outcome.kind == "unchanged"
                    else "published"
                ),
                **_context(outcome.context),
                change=outcome.kind,
                checkpoint_id=outcome.checkpoint_id,
                result=_value(outcome.current),
                previous=_value(outcome.previous),
                previous_evaluation=_value(outcome.previous_evaluation),
                evaluation=_value(outcome.evaluation),
            )
        elif isinstance(outcome, EvaluationResult):
            with self._lock:
                self._status = replace(self._status, evaluation=outcome.evaluation)
            self.record(
                "evaluation_completed",
                **_context(outcome.context),
                checkpoint_id=outcome.checkpoint_id,
                base_id=outcome.base.id,
                evaluation_id=evaluation_id,
                **asdict(outcome.evaluation),
            )

    def settings(self) -> Any:
        """Detached settings observed at initialization, if available."""
        with self._lock:
            return json.loads(json.dumps(self._settings))

    def final_settings(self) -> Any:
        """Detached participant settings captured before ownership release."""
        with self._lock:
            return json.loads(json.dumps(self._final_settings))

    def capture_final_settings(self) -> None:
        self.check()
        try:
            settings = self.redactor.snapshot(_value(self._final_settings_source()))
            with self._lock:
                self._final_settings = settings
        except Exception as exc:
            self._poison(exc)
            raise

    def finish(
        self,
        outcome: WorkerOutcome,
        *,
        report: tuple[dict[str, Any], Path | None] | None = None,
    ) -> str | None:
        """Called by the worker after resource release, even on failure."""
        if self.finished:
            raise RuntimeError("Journal is closed")
        rendered = None
        try:
            self.check()
            if outcome.error is not None:
                self.record(
                    "run_error",
                    error=f"{type(outcome.error).__name__}: {outcome.error}",
                    diagnostic=_value(outcome.error),
                )
            if outcome.released is not None:
                self.record("resources_released", **asdict(outcome.released))
            self.record(
                "run_finished",
                state=outcome.status.state.value,
                reason=outcome.status.reason,
            )
            if report is not None:
                data, path = report
                rendered = self._render_report(
                    {
                        **data,
                        "events": [json.loads(event.dump()) for event in self.events()],
                    },
                    path,
                )
        finally:
            # Close even a poisoned writer. Preserve the first failure if close
            # itself fails; never attempt another diagnostic write.
            try:
                if self._stream is not None:
                    self._stream.close()
            except Exception as exc:
                self._poison(exc)
            finally:
                self._stream = None
                with self._lock:
                    status = outcome.status
                    if self._error is not None:
                        primary = outcome.error or self._error
                        status = replace(
                            status,
                            state=UserState.FAILED,
                            reason=f"{type(primary).__name__}: {primary}",
                        )
                    self._status = replace(
                        status,
                        reason=self.redactor.text(status.reason)
                        if status.reason
                        else status.reason,
                    )
                self._done.set()
        if self.error is not None:
            raise self.error
        return rendered
