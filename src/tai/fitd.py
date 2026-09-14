"""A background User running the pinned FITD strategy."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, TextIO
from uuid import uuid4

from tai._profile import FITD_PROFILE
from tai.chats import ChatCancelled, LLMChat, ModelRole, _acquire_chats, _release_chats
from tai.evaluators import Evaluator
from tai.messages import Message
from tai.users import Evaluation, Progress, RunEvent, User, UserState, UserStatus


class FITDUser(User):
    """Independent FITD experiments with at most ten library rounds each.

    The presented text is the most recent Target response accepted for ladder
    progression in the retained conversation. Context kept for realignment is
    diagnostic until accepted. This composition policy is specific to FITDUser.
    """

    def __init__(
        self,
        target: LLMChat,
        attacker: LLMChat,
        base: Message,
        evaluator: Evaluator,
        *,
        artifacts_dir: str | Path = "runs",
    ) -> None:
        if target is attacker:
            raise ValueError(
                "Target and Attacker must be independent LLMChat instances"
            )
        self._target = target
        self._attacker = attacker
        self._base = base
        self._evaluator = evaluator
        self._artifacts_dir = Path(artifacts_dir)
        self._lock = Lock()
        self._execution = self._new_execution()

    def _new_execution(self) -> _Execution:
        return _Execution(
            self._target,
            self._attacker,
            self._base,
            self._evaluator,
            artifacts_dir=self._artifacts_dir,
        )

    def start(self) -> None:
        with self._lock:
            if self._execution.is_active():
                raise RuntimeError("User already has an active run")
            execution = self._new_execution()
            execution.start()
            self._execution = execution

    def stop(self) -> None:
        with self._lock:
            self._execution.stop()

    def status(self) -> UserStatus:
        with self._lock:
            return self._execution.status()

    def result(self) -> Message | None:
        with self._lock:
            return self._execution.result()

    def events(self, after: int = 0) -> tuple[RunEvent, ...]:
        with self._lock:
            return self._execution.events(after)

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            execution = self._execution
        return execution.wait(timeout)

    @staticmethod
    def _chat_settings(chat: LLMChat) -> dict[str, Any]:
        return chat.settings()


class _Execution:
    """One generation's state, callbacks, signals, and artifacts.

    Callbacks retain this execution, never the User's current execution. Even a
    retained callback cannot address a subsequent run's result or event journal.
    """

    def __init__(
        self,
        target: LLMChat,
        attacker: LLMChat,
        base: Message,
        evaluator: Evaluator,
        *,
        artifacts_dir: Path,
    ) -> None:
        self._participants: dict[ModelRole, tuple[LLMChat, str]] = {
            "Target": (target, "target_calls"),
            "Attacker": (attacker, "attacker_calls"),
        }
        self._chats = (target, attacker)
        self._base = base
        self._evaluator = evaluator
        self._artifacts_dir = Path(artifacts_dir)
        self._lock = Lock()
        self._status = UserStatus(
            progress=Progress(max_calls_per_conv=FITD_PROFILE.max_calls_per_conv)
        )
        self._result: Message | None = None
        self._events: list[RunEvent] = []
        self._flushed_events = 0
        self._worker: Thread | None = None
        self._stop_requested = Event()
        self._stop_revision = 0
        self._journal: TextIO | None = None
        self._run_dir: Path | None = None
        self._closed = False

    def start(self) -> None:
        with self._lock:
            _acquire_chats(self._chats, self)
            try:
                self._status = replace(
                    self._status, state=UserState.RUNNING, run_id=str(uuid4())
                )
                self._worker = Thread(
                    target=self._run, name=f"fitd-{self._status.run_id}"
                )
                self._worker.start()
            except Exception:
                _release_chats(self._chats, self)
                raise

    def is_active(self) -> bool:
        with self._lock:
            return self._status.state in (UserState.RUNNING, UserState.STOPPING) or (
                self._worker is not None and self._worker.is_alive()
            )

    def stop(self) -> None:
        with self._lock:
            if self._status.state is not UserState.RUNNING:
                return
            self._status = replace(self._status, state=UserState.STOPPING)
            # The same signal governs admission and the active generation, so a
            # stop between call_started and send cannot miss the next exchange.
            self._stop_requested.set()
            self._stop_revision += 1
            self._record_event("stop_requested")

    def status(self) -> UserStatus:
        with self._lock:
            return self._status

    def result(self) -> Message | None:
        with self._lock:
            return self._result

    def events(self, after: int = 0) -> tuple[RunEvent, ...]:
        with self._lock:
            return tuple(self._events[after:])

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            worker = self._worker
        if worker is None:
            return False
        worker.join(timeout)
        return not worker.is_alive()

    def _emit(self, kind: str, **data: Any) -> None:
        with self._lock:
            if self._closed:
                return
            self._record_event(kind, **data)
        self._flush_events()

    def _flush_events(self) -> None:
        with self._lock:
            pending = tuple(self._events[self._flushed_events :])
        # Only the worker writes artifacts; control methods never wait on I/O.
        if self._journal is not None:
            for event in pending:
                self._journal.write(event.dump() + "\n")
                self._journal.flush()
                self._flushed_events += 1

    def _record_event(self, kind: str, **data: Any) -> None:
        """Append under _lock; the worker flushes before its next action."""
        run_id = self._status.run_id
        assert run_id is not None
        event = RunEvent(
            run_id,
            len(self._events) + 1,
            kind,
            datetime.now(UTC).isoformat(),
            json.dumps(data, ensure_ascii=False),
        )
        self._events.append(event)

    def _send(self, role: ModelRole, message: Message) -> Message:
        chat, counter = self._participants[role]
        call_id = str(uuid4())
        with self._lock:
            # Admission is serialized with stop for both roles. Work already
            # admitted shares the stop signal even before entering chat.send.
            self._check_running()
            self._record_event(
                "call_started",
                role=role,
                call_id=call_id,
                input=json.loads(message.dump()),
            )
        self._flush_events()
        try:
            response = chat.send(message, cancellation=self._stop_requested)
        except ChatCancelled as exc:
            self._emit(
                "call_cancelled",
                role=role,
                call_id=call_id,
                response=json.loads(exc.response.dump())
                if exc.response is not None
                else None,
                **(
                    {"diagnostic": exc.diagnostic} if exc.diagnostic is not None else {}
                ),
            )
            raise
        except Exception as exc:
            diagnostic = getattr(exc, "diagnostic", None)
            self._emit(
                "call_failed",
                role=role,
                call_id=call_id,
                error=f"{type(exc).__name__}: {exc}",
                **({"diagnostic": diagnostic} if diagnostic is not None else {}),
            )
            raise
        # Persist each completed exchange before FITD does any further work.
        self._emit(
            "call_completed",
            role=role,
            call_id=call_id,
            response=json.loads(response.dump()),
        )
        with self._lock:
            progress = self._status.progress
            self._status = replace(
                self._status,
                progress=replace(progress, **{counter: getattr(progress, counter) + 1}),
            )
        return response

    def _checkpoint(self, role: ModelRole) -> object:
        self._check_running()
        return self._participants[role][0].checkpoint()

    def _restore(self, role: ModelRole, checkpoint: object) -> None:
        self._check_running()
        self._participants[role][0].restore(checkpoint)

    def _reset(self, role: ModelRole) -> None:
        self._check_running()
        self._participants[role][0].reset()

    def _publish(self, result: Message | None) -> None:
        with self._lock:
            if self._closed or self._result is result:
                return
            self._result = result
            self._status = replace(
                self._status,
                result_id=result.id if result is not None else None,
                evaluation=None,
            )
        assert self._run_dir is not None
        if result is None:
            self._emit("result_withdrawn")
            (self._run_dir / "result.txt").unlink(missing_ok=True)
            return
        self._emit("result_published", result=json.loads(result.dump()))
        (self._run_dir / "result.txt").write_text(result.to_string(), encoding="utf-8")

    def _check_running(self) -> None:
        if self._closed or self._stop_requested.is_set():
            raise ChatCancelled()

    def _evaluate(self) -> bool:
        self._check_running()
        result = self.result()
        if result is None:
            raise RuntimeError("FITD requested evaluation without a result")
        self._emit("evaluation_started", result_id=result.id)
        try:
            success = self._evaluator.is_success(result)
            if not isinstance(success, bool):
                raise TypeError("Evaluator.is_success must return bool")
        except Exception as exc:
            self._emit(
                "evaluation_failed",
                result_id=result.id,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        evaluation = Evaluation(result.id, success)
        self._emit("evaluation_completed", **asdict(evaluation))
        with self._lock:
            self._status = replace(self._status, evaluation=evaluation)
        return success

    def _run(self) -> None:
        run_id = self.status().run_id
        assert run_id is not None
        run_dir = self._artifacts_dir / run_id
        state, reason = UserState.COMPLETED, "round_limit"
        settings: dict[str, Any] | None = None
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            self._run_dir = run_dir
            with (run_dir / "events.jsonl").open("w", encoding="utf-8") as journal:
                self._journal = journal
                try:
                    for chat, _ in self._participants.values():
                        chat.reset()
                    from tai._fitd import GARAK_REVISION, run_fitd

                    settings = {
                        "garak_revision": GARAK_REVISION,
                        **asdict(FITD_PROFILE),
                        "base": json.loads(self._base.dump()),
                        **{
                            role: FITDUser._chat_settings(chat)
                            for role, (chat, _) in self._participants.items()
                        },
                    }
                    self._emit("run_started", settings=settings)
                    self._check_running()
                    reason = run_fitd(
                        self._base,
                        self._send,
                        self._publish,
                        self._evaluate,
                        self._check_running,
                        checkpoint=self._checkpoint,
                        restore=self._restore,
                        reset=self._reset,
                        emit=self._emit,
                    )
                except Exception as exc:
                    if isinstance(exc, ChatCancelled) and self._stop_requested.is_set():
                        state, reason = UserState.STOPPED, "user_stop"
                    else:
                        state, reason = UserState.FAILED, f"{type(exc).__name__}: {exc}"
                        self._emit("run_error", error=reason)
            self._journal = None
            self._finalize(run_dir, state, reason, settings)
        except Exception as exc:
            with self._lock:
                self._status = replace(
                    self._status,
                    state=UserState.FAILED,
                    reason=f"{type(exc).__name__}: {exc}",
                )
        finally:
            self._journal = None
            with self._lock:
                self._closed = True
            _release_chats(self._chats, self)

    def _finalize(
        self,
        run_dir: Path,
        state: UserState,
        reason: str,
        settings: dict[str, Any] | None,
    ) -> None:
        result = self.result()
        while True:
            with self._lock:
                revision = self._stop_revision
                if state is not UserState.FAILED and self._stop_requested.is_set():
                    state, reason = UserState.STOPPED, "user_stop"
                final_status = replace(self._status, state=state, reason=reason)
            with (run_dir / "events.jsonl").open("a", encoding="utf-8") as journal:
                self._journal = journal
                self._emit(
                    "run_finished", state=state.value, reason=reason, revision=revision
                )
                metadata = {
                    **asdict(final_status),
                    "finalization_revision": revision,
                    "result": json.loads(result.dump()) if result is not None else None,
                    "settings": settings,
                }
                (run_dir / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            self._journal = None
            with self._lock:
                # A stop during I/O invalidates the candidate. Append a newer
                # revision and rewrite metadata, without blocking control calls
                # or retracting anything from the append-only journal.
                if revision == self._stop_revision:
                    self._status = final_status
                    return
