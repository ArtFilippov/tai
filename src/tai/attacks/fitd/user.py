"""A background User running the pinned FITD strategy."""

from __future__ import annotations

from dataclasses import replace
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING
from uuid import uuid4

from tai.core.chats import LLMChat, ModelRole
from tai.core.chats.conversation import _acquire_chats, _release_chats
from tai.core.evaluation import Evaluator, evaluate
from tai.core.evaluation.assembly import EvaluatorSetup, evaluator_setup
from tai.core.primitives.messages import Message
from tai.core.runs import Evaluation, Progress, RunSnapshot, User, UserState, UserStatus
from tai.core.runs.operations import (
    Cause,
    Checkpoint,
    Context,
    ConversationChange,
    EvaluationResult,
    Exchange,
    ResourcesReleased,
    ResultChange,
    RunStarted,
    WorkerOutcome,
)

from .profile import FITD_PROFILE

if TYPE_CHECKING:
    from .garak_adapter import _FITD, _Target


class _RunStopped(Exception):
    """Admission of the next action was refused, not cancellation of a call."""


class FITDUser(User):
    """Independent FITD executions, without diagnostic observation.

    The presented text is the most recent Target response accepted for ladder
    progression in the retained conversation. Context kept for realignment is
    not a result. Use FITDExperiment to assemble this User with a run journal.
    """

    def __init__(
        self,
        target: LLMChat,
        attacker: LLMChat,
        base: Message,
        evaluator: Evaluator | EvaluatorSetup,
        *,
        max_calls_per_conv: int = 10,
    ) -> None:
        self._evaluation_setup = evaluator_setup(evaluator)
        chats: dict[ModelRole, LLMChat] = {"Target": target, "Attacker": attacker}
        if self._evaluation_setup.chat is not None:
            chats["Evaluator"] = self._evaluation_setup.chat
        if len({id(chat) for chat in chats.values()}) != len(chats):
            raise ValueError("Participants must use independent LLMChat instances")
        self._source_chats = chats
        self._target = target
        self._attacker = attacker
        self._base = base
        self._evaluator = self._evaluation_setup.evaluator
        if type(max_calls_per_conv) is not int or max_calls_per_conv <= 0:
            raise ValueError("max_calls_per_conv must be a positive integer")
        self._profile = replace(FITD_PROFILE, max_calls_per_conv=max_calls_per_conv)
        self._lock = Lock()
        self._execution = self._new_execution()

    def _new_execution(
        self, execution_type: type[_Execution] | None = None
    ) -> _Execution:
        return (execution_type or _Execution)(
            self._source_chats,
            self._base,
            self._evaluator,
            max_calls_per_conv=self._profile.max_calls_per_conv,
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

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            execution = self._execution
        return execution.wait(timeout)

    def snapshot(self) -> RunSnapshot:
        with self._lock:
            return self._execution.snapshot()


class _Execution:
    """One generation's work and ownership. Callbacks cannot address a later run."""

    def __init__(
        self,
        chats: dict[ModelRole, LLMChat],
        base: Message,
        evaluator: Evaluator,
        *,
        max_calls_per_conv: int,
    ) -> None:
        self._participants = {
            role: (chat, role.lower() + "_calls") for role, chat in chats.items()
        }
        self._chats = tuple(chat for chat, _ in self._participants.values())
        self._base = base
        self._evaluator = evaluator
        self._profile = replace(FITD_PROFILE, max_calls_per_conv=max_calls_per_conv)
        self._lock = Lock()
        self._status = UserStatus(
            progress=Progress(max_calls_per_conv=max_calls_per_conv)
        )
        self._result: Message | None = None
        self._worker: Thread | None = None
        self._stop_requested = Event()
        self._closed = False

    def start(self) -> RunStarted:
        with self._lock:
            _acquire_chats(self._chats, self)
            try:
                run_id = str(uuid4())
                self._status = replace(
                    self._status, state=UserState.RUNNING, run_id=run_id
                )
                self._admit(RunStarted(run_id))
                self._worker = Thread(
                    target=self._run, args=(run_id,), name=f"fitd-{run_id}"
                )
                self._worker.start()
            except Exception:
                _release_chats(self._chats, self)
                raise
            return RunStarted(run_id)

    def _admit(self, started: RunStarted) -> None:
        """Admission boundary before the worker can execute or stop can return."""

    def is_active(self) -> bool:
        with self._lock:
            return self._status.state in (UserState.RUNNING, UserState.STOPPING) or (
                self._worker is not None and self._worker.is_alive()
            )

    def stop(self) -> bool:
        with self._lock:
            return self._request_stop()

    def _request_stop(self) -> bool:
        """Apply control under the admission lock; wrappers must not do I/O here."""
        if self._status.state is not UserState.RUNNING:
            return False
        self._status = replace(self._status, state=UserState.STOPPING)
        self._stop_requested.set()
        return True

    def status(self) -> UserStatus:
        with self._lock:
            return self._status

    def result(self) -> Message | None:
        with self._lock:
            return self._result

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            worker = self._worker
        if worker is None:
            return False
        worker.join(timeout)
        return not worker.is_alive()

    def snapshot(self) -> RunSnapshot:
        with self._lock:
            if self._worker is None or self._worker.is_alive():
                raise RuntimeError("Snapshot requires a finished worker; wait first")
            return RunSnapshot(self._status, self._result)

    def _send(
        self,
        role: ModelRole,
        message: Message,
        cause: Cause,
    ) -> Exchange:
        with self._lock:
            self._ensure_running()
        return self._exchange(self._context(role, cause), str(uuid4()), message)

    def _exchange(
        self,
        context: Context,
        call_id: str,
        message: Message,
    ) -> Exchange:
        """An admitted call completes even if stop is requested before model I/O."""
        chat, counter = self._participants[context.role]
        response = chat.send(message)
        with self._lock:
            progress = self._status.progress
            self._status = replace(
                self._status,
                progress=replace(progress, **{counter: getattr(progress, counter) + 1}),
            )
        return Exchange(context, call_id, message, response)

    def _context(self, role: ModelRole, cause: Cause) -> Context:
        run_id = self._status.run_id
        assert run_id is not None
        return Context(run_id, role, self._conversation_id(role), cause)

    def _checkpoint(self, role: ModelRole, cause: Cause) -> Checkpoint:
        with self._lock:
            self._ensure_open()
        token = self._participants[role][0].checkpoint()
        return Checkpoint(self._context(role, cause), str(uuid4()), token)

    def _restore(
        self, role: ModelRole, checkpoint: Checkpoint, cause: Cause
    ) -> ConversationChange:
        self._check_running()
        self._participants[role][0].restore(checkpoint.token)
        context = self._context(role, cause)
        return ConversationChange(
            "restored", context, context.conversation_id, checkpoint
        )

    def _reset(
        self,
        role: ModelRole,
        cause: Cause,
    ) -> ConversationChange:
        with self._lock:
            self._ensure_running()
        previous = self._conversation_id(role)
        self._participants[role][0].reset()
        return ConversationChange("reset", self._context(role, cause), previous)

    def _conversation_id(self, role: ModelRole) -> str:
        return self._participants[role][0].conversation_id

    def _publish(
        self, result: Message | None, context: Context, checkpoint_id: str
    ) -> ResultChange:
        with self._lock:
            previous, evaluation = self._result, self._status.evaluation
            if self._closed or self._result is result:
                return ResultChange(
                    "unchanged",
                    context,
                    checkpoint_id,
                    previous,
                    previous,
                    evaluation,
                    evaluation,
                )
            self._ensure_running()
            self._result = result
            self._status = replace(
                self._status,
                result_id=result.id if result is not None else None,
                evaluation=None,
            )
            change = ResultChange(
                "withdrawn"
                if result is None
                else ("accepted" if previous is None else "replaced"),
                context,
                checkpoint_id,
                previous,
                result,
                evaluation,
                None,
            )
        return change

    def _check_running(self) -> None:
        with self._lock:
            self._ensure_running()

    def _ensure_running(self) -> None:
        self._ensure_open()
        if self._stop_requested.is_set():
            raise _RunStopped()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Execution is closed")

    def _evaluate(
        self, result: Message, context: Context, checkpoint_id: str
    ) -> EvaluationResult:
        with self._lock:
            self._ensure_running()
            if self._result is not result:
                raise RuntimeError(
                    "FITD requested evaluation of an unpresented Message"
                )
            self._status = replace(
                self._status,
                evaluation=None,
                progress=replace(
                    self._status.progress,
                    evaluator_calls=self._status.progress.evaluator_calls + 1,
                ),
            )
        return self._assess(result, context, checkpoint_id)

    def _assess(
        self, result: Message, context: Context, checkpoint_id: str
    ) -> EvaluationResult:
        success = evaluate(self._evaluator, self._base, result)
        return self._record_evaluation(result, success, context, checkpoint_id)

    def _record_evaluation(
        self, result: Message, success: bool, context: Context, checkpoint_id: str
    ) -> EvaluationResult:
        evaluation = Evaluation(result.id, success)
        with self._lock:
            if self._closed or self._result is not result:
                raise RuntimeError("Cannot record evaluation of an unpresented Message")
            self._status = replace(self._status, evaluation=evaluation)
        return EvaluationResult(context, checkpoint_id, result, evaluation, self._base)

    def _initialize(self, run_id: str) -> None:
        for role in self._participants:
            self._reset(role, Cause("run_started"))

    def _run_strategy(
        self,
        *,
        target_type: type[_Target] | None = None,
        probe_type: type[_FITD] | None = None,
    ) -> str:
        from .garak_adapter import _FITD, _Target, run_fitd

        self._check_running()
        return run_fitd(
            self._base,
            self._send,
            self._publish,
            self._evaluate,
            self._check_running,
            checkpoint=self._checkpoint,
            restore=self._restore,
            reset=self._reset,
            conversation_id=self._conversation_id,
            profile=self._profile,
            target_type=target_type or _Target,
            probe_type=probe_type or _FITD,
        )

    def _run(self, run_id: str) -> WorkerOutcome:
        state, reason = UserState.COMPLETED, "round_limit"
        error: Exception | None = None
        released: ResourcesReleased | None = None
        try:
            self._initialize(run_id)
            reason = self._run_strategy()
        except _RunStopped:
            state, reason = UserState.STOPPED, "user_stop"
        except Exception as exc:
            error = exc
            state, reason = UserState.FAILED, f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self._finalize(state, reason)
            except Exception as exc:
                error = error or exc
                self._fail(error)
            with self._lock:
                self._closed = True
            try:
                released = self._release_resources(run_id)
            except Exception as exc:
                error = error or exc
                self._fail(error)
        outcome = WorkerOutcome(self.status(), self.result(), error, released)
        try:
            self._complete(outcome)
        except Exception as exc:
            error = error or exc
            self._fail(error)
            outcome = WorkerOutcome(self.status(), self.result(), error, released)
        return outcome

    def _fail(self, error: Exception) -> None:
        with self._lock:
            self._status = replace(
                self._status,
                state=UserState.FAILED,
                reason=f"{type(error).__name__}: {error}",
            )

    def _release_resources(self, run_id: str) -> ResourcesReleased:
        _release_chats(self._chats, self)
        return ResourcesReleased(run_id, tuple(self._participants))

    def _finalize(self, state: UserState, reason: str) -> None:
        with self._lock:
            if state is not UserState.FAILED and self._stop_requested.is_set():
                state, reason = UserState.STOPPED, "user_stop"
            self._status = replace(self._status, state=state, reason=reason)

    def _complete(self, outcome: WorkerOutcome) -> None:
        """Worker completion boundary after finalization and participant release."""
