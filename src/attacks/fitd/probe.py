"""Pinned garak integration. No library types cross the runtime port.

FITD retains its initial-turn builder, ladder builder, adaptive branch logic,
and IterativeProbe loop. Overrides bind task, model I/O, evaluation, and reporting
to this run. No global configuration or plugin registry is modified.

The public façade owns a file-local native extension. Native decisions dispatch
through the façade so observers intercept the same operations without exposing
garak's framework setup methods.
"""

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import override
from uuid import uuid4

from core.primitives.messages import Message
from core.runs.operations import Cause, Context, Exchange, LocalDecision

from .contracts import CompletionReason, Continuation, ProbeOperations, Selection, StrategyPorts
from .continuation import CONTINUE, SELECT_CONTINUATION, branch_kind, describe_selection
from .native import Attempt, Conversation, FITD, attempt_prompt, levels_completed
from .native_configuration import configure, prepare_goal, prepare_ladder
from .profile import EnglishLanguage, FITDProfile
from .target import FITDTarget, history

__all__ = ["FITDProbe"]


@dataclass
class _ProbeState:
    last_attacker_exchange: Exchange | None = None
    attempt_id: str | None = None
    decision_cause: Cause | None = None
    checks: list[LocalDecision] = field(default_factory=list[LocalDecision])
    requests: dict[str, Message] = field(default_factory=dict[str, Message])
    success: bool = False
    failure: Exception | None = None


class _NativeFITD(FITD):
    def __init__(
        self, base: Message, ports: StrategyPorts, target: FITDTarget, *,
        profile: FITDProfile, operations: ProbeOperations[Conversation, Attempt],
    ) -> None:
        self._profile, self._base, self._ports, self._target = profile, base, ports, target
        self._state = _ProbeState()
        self._language = EnglishLanguage()
        self._operations = operations
        super().__init__(config_root={})

    def raise_if_interrupted(self) -> None:
        if self._state.failure is not None:
            raise self._state.failure
        self._ports.check_running()

    @contextmanager
    def _guard_execution(self) -> Generator[None]:
        # Retain the original error even if the native loop catches/replaces it.
        self._operations.raise_if_interrupted()
        try:
            yield
        except Exception as exc:
            if self._state.failure is None:
                self._state.failure = exc
            raise
        self._operations.raise_if_interrupted()

    @override
    def _load_config(self, config_root: object = None) -> None:
        # Bind all execution settings locally instead of loading global config.
        configure(self, self._profile, config_root)

    @override
    def _get_langprovider(self) -> EnglishLanguage:
        return self._language

    @override
    def _get_reverse_langprovider(self) -> EnglishLanguage:
        return self._language

    @override
    def _buff_hook(self, attempts: Iterable[Attempt]) -> Iterable[Attempt]:
        self._operations.raise_if_interrupted()
        return attempts

    @override
    def _get_red_team_response(self, prompt: str) -> str:
        # Native red-team requests are independent one-turn conversations.
        with self._guard_execution():
            cause = self._state.decision_cause or Cause("prepare_attack", self._state.attempt_id)
            conversation = self._ports.conversation
            _ = conversation.reset("Attacker", cause)
            exchange = self._ports.send("Attacker", self._base.spawn(prompt), cause)
            self._state.last_attacker_exchange = exchange
            response = exchange.response
            return response.to_string()

    @override
    def _is_refusal(self, response: str) -> bool:
        if self._state.attempt_id is None:
            exchange = self._state.last_attacker_exchange
            assert exchange is not None
            decision = self._operations.check_refusal(exchange.request, exchange.response, exchange.context, None)
        else:
            point = self._target.current
            assert point.request is not None and point.response is not None
            assert self._state.decision_cause is not None
            decision = self._operations.check_refusal(
                point.request, point.response,
                replace(point.checkpoint.context, cause=self._state.decision_cause), point.checkpoint.id,
            )
        self._state.checks.append(decision)
        assert decision.response.to_string() == response
        return decision.value

    def check_refusal(
        self, request: Message, response: Message, context: Context, checkpoint_id: str | None,
    ) -> LocalDecision:
        self._operations.raise_if_interrupted()
        value = super()._is_refusal(response.to_string())
        return LocalDecision("refusal", value, context, request, response, checkpoint_id)

    @override
    def _is_response_aligned(self, request: str, response: str) -> bool:
        point = self._target.current
        assert point.request is not None and point.response is not None
        assert self._state.decision_cause is not None
        request_message, response_message = point.request, point.response
        assert request_message.to_string() == request and response_message.to_string() == response
        decision = self._operations.check_alignment(
            point.request, point.response,
            replace(point.checkpoint.context, cause=self._state.decision_cause), point.checkpoint.id,
        )
        self._state.checks.append(decision)
        return decision.value

    def check_alignment(
        self, request: Message, response: Message, context: Context, checkpoint_id: str,
    ) -> LocalDecision:
        self._operations.raise_if_interrupted()
        value = super()._is_response_aligned(request.to_string(), response.to_string())
        judgment = self._state.last_attacker_exchange
        assert judgment is not None
        return LocalDecision("alignment", value, context, request, response, checkpoint_id, judgment)

    @override
    def _create_init_attempts(self) -> list[Attempt]:
        with self._guard_execution():
            goal = self._base.to_string()
            prepare_goal(self, goal)
            initial = self._generate_init_turn(goal)
            self._operations.raise_if_interrupted()
            if initial is None:
                return []
            prepare_ladder(self, goal, self._get_level_queries(goal, initial))
            self._operations.raise_if_interrupted()
            attempt = self._create_attempt(initial)
            self._state.requests[str(attempt.uuid)] = self._base.spawn(initial)
            attempt.notes.update({
                "turn_num": 0, "target_turn": goal, "levels_completed": 0,
                "is_adversarial": False, "is_last_turn": False, "last_non_refused_prompt": None,
            })
            return [attempt]

    @override
    def _verify_is_last_response_success(self, attempt: Attempt) -> list[bool]:
        # Native verification precedes acceptance. Our evaluation occurs after
        # adaptation so rejected material cannot earn a successful result.
        self._operations.raise_if_interrupted()
        assert levels_completed(attempt) >= 0
        return [False]

    @override
    def _generate_next_attempts(self, last_attempt: Attempt) -> list[Attempt]:
        with self._guard_execution():
            self._state.attempt_id = str(last_attempt.uuid)
            try:
                return super()._generate_next_attempts(last_attempt)
            finally:
                self._state.attempt_id = None

    def _accept(self, last_attempt: Attempt, cause: Cause) -> bool:
        accepted = self._target.accept(cause)
        if last_attempt.notes["is_adversarial"]:
            assert accepted.current is not None
            evaluation = self._ports.evaluate(accepted.current, accepted.context, accepted.checkpoint_id)
            self._operations.raise_if_interrupted()
            self._state.success = evaluation.evaluation.success
        return self._state.success

    @override
    def generate_next_turn_attempt(self, conversation: Conversation, last_attempt: Attempt) -> Attempt | None:
        return self._operations.generate_next_turn_attempt(conversation, last_attempt)

    def advance(self, conversation: Conversation, last_attempt: Attempt) -> Attempt | None:
        continuation = self._operations.select_continuation(conversation, last_attempt)
        self._operations.raise_if_interrupted()
        decision, next_attempt = continuation.decision, continuation.attempt
        cause = replace(decision.context.cause, reason=decision.branch)
        if decision.branch == CONTINUE and self._accept(last_attempt, cause):
            return None
        if next_attempt is not None:
            _ = self._target.synchronize(attempt_prompt(next_attempt), cause)
        return next_attempt

    def select_continuation(self, conversation: Conversation, last_attempt: Attempt) -> Continuation[Attempt]:
        self._operations.raise_if_interrupted()
        cause = Cause(SELECT_CONTINUATION, str(last_attempt.uuid), str(uuid4()))
        self._state.decision_cause, self._state.checks = cause, []
        try:
            next_attempt = super().generate_next_turn_attempt(conversation, last_attempt)
        finally:
            self._state.decision_cause = None
        point = self._target.current
        assert point.response is not None
        next_request = None
        if next_attempt is not None:
            next_request = self._base.spawn(history(attempt_prompt(next_attempt))[-1][1])
            self._state.requests[str(next_attempt.uuid)] = next_request
        selection = Selection(cause, point, tuple(self._state.checks))
        return describe_selection(
            selection, (last_attempt, next_attempt), branch_kind(conversation, last_attempt, next_attempt), next_request,
        )

    @override
    def _execute_attempt(self, this_attempt: Attempt) -> Attempt:
        with (
            self._guard_execution(),
            self._target.attempt(str(this_attempt.uuid), self._state.requests.pop(str(this_attempt.uuid))),
        ):
            return super()._execute_attempt(this_attempt)

    @override
    def _execute_all(self, attempts: Iterable[Attempt]) -> list[Attempt]:
        # Avoid native global reportfile writes, retaining execution/postprocess.
        with self._guard_execution():
            return [self._postprocess_attempt(self._execute_attempt(attempt)) for attempt in attempts]

    def completion_reason(self) -> CompletionReason:
        self._operations.raise_if_interrupted()
        if self._state.success:
            return "evaluator_success"
        return "round_limit" if self.attempt_queue else "strategy_exhausted"

    def execute(self) -> CompletionReason:
        _ = self.probe(self._target)
        # The native loop may catch GarakException; retain the boundary's outcome.
        return self._operations.completion_reason()


class FITDProbe(ProbeOperations[Conversation, Attempt]):
    def __init__(
        self, base: Message, ports: StrategyPorts, target: FITDTarget, *, profile: FITDProfile,
    ) -> None:
        super().__init__()
        self._native = _NativeFITD(base, ports, target, profile=profile, operations=self)

    @override
    def __call__(self) -> CompletionReason:
        return self._native.execute()

    @override
    def raise_if_interrupted(self) -> None:
        self._native.raise_if_interrupted()

    @override
    def check_refusal(
        self, request: Message, response: Message, context: Context, checkpoint_id: str | None,
    ) -> LocalDecision:
        return self._native.check_refusal(request, response, context, checkpoint_id)

    @override
    def check_alignment(
        self, request: Message, response: Message, context: Context, checkpoint_id: str,
    ) -> LocalDecision:
        return self._native.check_alignment(request, response, context, checkpoint_id)

    @override
    def select_continuation(self, conversation: Conversation, last_attempt: Attempt) -> Continuation[Attempt]:
        return self._native.select_continuation(conversation, last_attempt)

    @override
    def generate_next_turn_attempt(self, conversation: Conversation, last_attempt: Attempt) -> Attempt | None:
        return self._native.advance(conversation, last_attempt)

    @override
    def completion_reason(self) -> CompletionReason:
        return self._native.completion_reason()
