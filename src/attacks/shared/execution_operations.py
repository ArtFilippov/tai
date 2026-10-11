"""Admitted conversation, presentation, and evaluation operations for one run."""

from dataclasses import replace
from threading import Lock
from typing import Literal
from uuid import uuid4

from core.evaluation import evaluate
from core.primitives.messages import Message
from core.runs import Evaluation, Progress, UserStatus
from core.runs.operations import (
    Cause, Checkpoint, Context, ConversationChange, EvaluationResult, Exchange, ResultChange,
)

from .execution_state import ExecutionSettings, ExecutionState, RunStopped

type ModelRole = Literal["Target", "Attacker", "Evaluator"]


class ExecutionOperations:
    def __init__(self, settings: ExecutionSettings) -> None:
        super().__init__()
        self._settings = settings
        self._state = ExecutionState(UserStatus(progress=Progress(max_calls_per_conv=settings.configuration.max_calls_per_conv)))
        self._lock = Lock()

    @property
    def base(self) -> Message:
        return self._settings.base

    def _send(self, role: ModelRole, message: Message, cause: Cause) -> Exchange:
        self._check_running()
        return self.exchange(self._context(role, cause), str(uuid4()), message)

    def exchange(self, context: Context, call_id: str, message: Message) -> Exchange:
        """Complete an admitted call even after a stop request.

        Args:
            context (Context): Identity and cause of the admitted exchange.
            call_id (str): Unique call identifier.
            message (Message): The request to send.

        Returns:
            Exchange: The completed request and response.
        """
        response = self._settings.chats[context.role].send(message)
        with self._lock:
            progress = self._state.status.progress
            counters = {
                "Target": {"target_calls": progress.target_calls + 1},
                "Attacker": {"attacker_calls": progress.attacker_calls + 1},
                "Evaluator": {"evaluator_calls": progress.evaluator_calls + 1},
            }
            self._state.status = replace(self._state.status, progress=replace(progress, **counters[context.role]))
        return Exchange(context, call_id, message, response)

    def _context(self, role: ModelRole, cause: Cause) -> Context:
        run_id = self._state.status.run_id
        assert run_id is not None
        return Context(run_id, role, self._conversation_id(role), cause)

    def checkpoint(self, role: ModelRole, cause: Cause) -> Checkpoint:
        with self._lock:
            self.ensure_open()
        token = self._settings.chats[role].checkpoint()
        return Checkpoint(self._context(role, cause), str(uuid4()), token)

    def restore(self, role: ModelRole, checkpoint: Checkpoint, cause: Cause) -> ConversationChange:
        self._check_running()
        self._settings.chats[role].restore(checkpoint.token)
        context = self._context(role, cause)
        return ConversationChange("restored", context, context.conversation_id, checkpoint)

    def reset(self, role: ModelRole, cause: Cause) -> ConversationChange:
        self._check_running()
        previous = self._conversation_id(role)
        self._settings.chats[role].reset()
        return ConversationChange("reset", self._context(role, cause), previous)

    def _conversation_id(self, role: ModelRole) -> str:
        return self._settings.chats[role].conversation_id

    def publish(self, result: Message | None, context: Context, checkpoint_id: str) -> ResultChange:
        with self._lock:
            previous, evaluation = self._state.result, self._state.status.evaluation
            if self._state.closed or previous is result:
                return ResultChange("unchanged", context, checkpoint_id, previous, previous, evaluation, evaluation)
            self._ensure_running()
            self._state.result = result
            self._state.status = replace(self._state.status, result_id=result.id if result is not None else None, evaluation=None)
            change = ResultChange(
                "withdrawn" if result is None else ("accepted" if previous is None else "replaced"),
                context, checkpoint_id, previous, result, evaluation, None,
            )
        return change

    def _check_running(self) -> None:
        with self._lock:
            self._ensure_running()

    def _ensure_running(self) -> None:
        self.ensure_open()
        if self._state.stopping:
            raise RunStopped()

    def ensure_open(self) -> None:
        if self._state.closed:
            raise RuntimeError("Execution is closed")

    def _evaluate(self, result: Message, context: Context, checkpoint_id: str) -> EvaluationResult:
        with self._lock:
            self._ensure_running()
            if self._state.result is not result:
                raise RuntimeError("Strategy requested evaluation of an unpresented Message")
            self._state.status = replace(
                self._state.status, evaluation=None,
                progress=replace(self._state.status.progress, evaluator_calls=self._state.status.progress.evaluator_calls + 1),
            )
        return self.assess(result, context, checkpoint_id)

    def assess(self, result: Message, context: Context, checkpoint_id: str) -> EvaluationResult:
        success = evaluate(self._settings.evaluator, self.base, result)
        return self.record_evaluation(result, success, context, checkpoint_id)

    def record_evaluation(self, result: Message, success: bool, context: Context, checkpoint_id: str) -> EvaluationResult:
        evaluation = Evaluation(result.id, success)
        with self._lock:
            if self._state.closed or self._state.result is not result:
                raise RuntimeError("Cannot record evaluation of an unpresented Message")
            self._state.status = replace(self._state.status, evaluation=evaluation)
        return EvaluationResult(context, checkpoint_id, result, evaluation, self.base)
