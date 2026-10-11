"""Lifecycle and resource ownership of an independent strategy execution."""

from dataclasses import replace
from threading import Thread
from types import TracebackType
from uuid import uuid4

from core.chats.reservation_utils import acquire_chats, release_chats
from core.primitives.messages import Message
from core.runs import RunSnapshot, UserState, UserStatus
from core.runs.operations import Cause, ResourcesReleased, RunStarted, WorkerOutcome

from .contracts import CompletionReason, ConversationPorts, StrategyPorts
from .execution_operations import ExecutionOperations
from .execution_state import RunStopped


class _CaptureFailure:
    def __init__(self, errors: list[Exception]) -> None:
        super().__init__()
        self._errors = errors

    def __enter__(self) -> None:
        return None

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None,
    ) -> bool:
        # Worker-owned exception reporting still allows all cleanup phases to run.
        if isinstance(error, Exception):
            self._errors.append(error)
            return True
        return False


class Execution(ExecutionOperations):
    """One generation's work and ownership. Callbacks cannot address a later run."""

    def start(self) -> RunStarted:
        with self._lock:
            acquire_chats(tuple(self._settings.chats.values()), self)
            try:
                started = self._start_worker()
            except Exception:
                release_chats(tuple(self._settings.chats.values()), self)
                raise
            return started

    def _start_worker(self) -> RunStarted:
        run_id = str(uuid4())
        self._state.status = replace(self._state.status, state=UserState.RUNNING, run_id=run_id)
        self.admit(RunStarted(run_id))
        self._state.start_worker(Thread(target=self._run, args=(run_id,), name=f"attack-{run_id}"))
        return RunStarted(run_id)

    def admit(self, started: RunStarted) -> None:
        """Check admission before the worker can execute or stop can return.

        Args:
            started (RunStarted): The newly reserved run.

        Raises:
            RuntimeError: The admission does not belong to this execution.
        """
        if started.run_id != self._state.status.run_id:
            raise RuntimeError("Admission belongs to a different execution")

    def is_active(self) -> bool:
        with self._lock:
            return self._state.status.state in (UserState.RUNNING, UserState.STOPPING) or self._state.working

    def stop(self) -> bool:
        with self._lock:
            return self.request_stop()

    def request_stop(self) -> bool:
        """Apply control under the admission lock; wrappers must not do I/O here.

        Returns:
            bool: Whether this request changed the admission state.
        """
        if self._state.status.state is not UserState.RUNNING:
            return False
        self._state.status = replace(self._state.status, state=UserState.STOPPING)
        self._state.request_stop()
        return True

    def status(self) -> UserStatus:
        with self._lock:
            return self._state.status

    def result(self) -> Message | None:
        with self._lock:
            return self._state.result

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            worker = self._state.worker
        if worker is None:
            return False
        worker.join(timeout)
        return not worker.is_alive()

    def snapshot(self) -> RunSnapshot:
        with self._lock:
            if self._state.worker is None or self._state.working:
                raise RuntimeError("Snapshot requires a finished worker; wait first")
            return RunSnapshot(self._state.status, self._state.result)

    def initialize(self, run_id: str) -> None:
        assert run_id == self._state.status.run_id
        for role in self._settings.chats:
            _ = self.reset(role, Cause("run_started"))

    def _run_strategy(self) -> CompletionReason:
        self._check_running()
        ports = StrategyPorts(
            self._send, self.publish, self._evaluate, self._check_running,
            ConversationPorts(self.checkpoint, self.restore, self.reset, self._conversation_id),
        )
        return self._settings.runtime(self.base, ports)

    def _execute(self, run_id: str) -> tuple[UserState, str]:
        self.initialize(run_id)
        return UserState.COMPLETED, self._run_strategy()

    def _work(self, run_id: str) -> tuple[UserState, str]:
        try:
            return self._execute(run_id)
        except RunStopped:
            return UserState.STOPPED, "user_stop"

    def _run(self, run_id: str) -> WorkerOutcome:
        errors: list[Exception] = []
        state, reason = UserState.COMPLETED, "round_limit"
        with _CaptureFailure(errors):
            state, reason = self._work(run_id)
        self._reflect_failure(errors)
        if errors:
            state, reason = UserState.FAILED, f"{type(errors[0]).__name__}: {errors[0]}"
        return self._finish_work(run_id, (state, reason), errors)

    def _finish_work(self, run_id: str, terminal: tuple[UserState, str], errors: list[Exception]) -> WorkerOutcome:
        released: ResourcesReleased | None = None
        with _CaptureFailure(errors):
            self.finalize(*terminal)
        with self._lock:
            self._state.closed = True
        with _CaptureFailure(errors):
            released = self._release_resources(run_id)
        self._reflect_failure(errors)
        outcome = WorkerOutcome(self.status(), self.result(), errors[0] if errors else None, released)
        with _CaptureFailure(errors):
            self.complete(outcome)
        self._reflect_failure(errors)
        return WorkerOutcome(self.status(), self.result(), errors[0] if errors else None, released)

    def _reflect_failure(self, errors: list[Exception]) -> None:
        if errors:
            with self._lock:
                self._state.status = replace(self._state.status, state=UserState.FAILED, reason=f"{type(errors[0]).__name__}: {errors[0]}")

    def _release_resources(self, run_id: str) -> ResourcesReleased:
        release_chats(tuple(self._settings.chats.values()), self)
        return ResourcesReleased(run_id, tuple(self._settings.chats))

    def finalize(self, state: UserState, reason: str) -> None:
        with self._lock:
            if state is not UserState.FAILED and self._state.stopping:
                state, reason = UserState.STOPPED, "user_stop"
            self._state.status = replace(self._state.status, state=state, reason=reason)

    def complete(self, outcome: WorkerOutcome) -> None:
        """Validate completion after finalization and participant release.

        Args:
            outcome (WorkerOutcome): The final outcome including cleanup failures.

        Raises:
            RuntimeError: The completion does not belong to this execution.
        """
        if outcome.status.run_id != self._state.status.run_id:
            raise RuntimeError("Completion belongs to a different execution")
