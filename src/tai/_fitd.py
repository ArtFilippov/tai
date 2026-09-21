"""Pinned garak integration. No library types cross this module's boundary.

FITD retains its initial-turn builder, ladder builder, adaptive branch logic,
and IterativeProbe loop. Overrides bind task, model I/O, evaluation, and reporting
to this run. No global configuration or plugin registry is modified.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Literal
from uuid import uuid4

from garak.attempt import Attempt, Conversation
from garak.attempt import Message as GarakMessage
from garak.detectors.mitigation import Prefixes
from garak.probes.fitd import FITD

from tai._operations import (
    BranchDecision,
    Cause,
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
)
from tai._profile import FITD_PROFILE, _FITDProfile
from tai.chats import ModelRole
from tai.messages import Message

GARAK_REVISION = "3f50ea5ff9cd7050099940647c15c39b07a93392"

type _Send = Callable[[ModelRole, Message, Cause], Exchange]
type _Checkpoint = Callable[[ModelRole, Cause], Checkpoint]
type _Restore = Callable[[ModelRole, Checkpoint, Cause], ConversationChange]
type _Reset = Callable[[ModelRole, Cause], ConversationChange]
type _Publish = Callable[[Message | None, Context, str], ResultChange]
type _Evaluate = Callable[[Message, Context, str], EvaluationResult]
type _ConversationId = Callable[[ModelRole], str]
type _Emit = Callable[..., None]
type _History = tuple[tuple[str, str], ...]


class _English:
    """The supported English profile needs no translation service or models."""

    target_lang = "en"

    def get_text(self, prompts):
        return prompts


@dataclass(frozen=True)
class _Continuation:
    # The native value never leaves the adapter. Wrappers use decision alone.
    decision: BranchDecision
    attempt: Attempt | None


def _history(conversation: Conversation) -> _History:
    return tuple((turn.role, turn.content.text) for turn in conversation.turns)


class _Target:
    """Map library prefixes to chat-owned snapshots, never provider histories.

    A completed response is diagnostic until FITD chooses to advance its ladder.
    Realignment retains off-topic context without presenting it as a result.
    """

    def __init__(
        self,
        send: _Send,
        publish: _Publish,
        checkpoint: _Checkpoint,
        restore: _Restore,
        reset: _Reset,
        emit: _Emit,
        conversation_id: _ConversationId,
    ) -> None:
        self._send = send
        self._publish = publish
        self._checkpoint = checkpoint
        self._restore = restore
        self._reset = reset
        self._emit = emit
        self._history: _History = ()
        self._snapshots: dict[_History, ConversationPoint] = {}
        self._get_conversation_id = conversation_id
        self._conversation_id = conversation_id("Target")
        self._attempt_id: str | None = None
        self._request: Message | None = None
        self.save(None, None, None, Cause("initial_checkpoint"))

    @property
    def current(self) -> ConversationPoint:
        return self._snapshots[self._history]

    @contextmanager
    def attempt(self, attempt_id: str, request: Message) -> Iterator[None]:
        self._attempt_id = attempt_id
        self._request = request
        try:
            yield
        finally:
            self._attempt_id = None
            self._request = None

    def save(
        self,
        request: Message | None,
        response: Message | None,
        result: Message | None,
        cause: Cause,
    ) -> ConversationPoint:
        snapshot = ConversationPoint(
            self._checkpoint("Target", cause), request, response, result
        )
        self._snapshots[self._history] = snapshot
        self._emit(
            "checkpoint_created",
            role="Target",
            conversation_id=self._conversation_id,
            checkpoint_id=snapshot.checkpoint.id,
            response_id=response.id if response is not None else None,
            result_id=result.id if result is not None else None,
        )
        return snapshot

    def synchronize(self, prompt: Conversation, cause: Cause) -> Synchronization:
        turns = _history(prompt)
        if not turns or turns[-1][0] != "user":
            raise RuntimeError("FITD must supply a final user turn")
        prefix = turns[:-1]
        previous = self.current
        context = replace(previous.checkpoint.context, cause=cause)
        if prefix == self._history:
            change = ConversationChange(
                "unchanged", context, self._conversation_id, previous.checkpoint
            )
        elif not prefix:
            change = self._reset("Target", cause)
            old_conversation_id = self._conversation_id
            self._conversation_id = self._get_conversation_id("Target")
            self._history = ()
            self._snapshots.clear()
            self._emit(
                "history_reset",
                role="Target",
                conversation_id=self._conversation_id,
                previous_conversation_id=old_conversation_id,
                from_checkpoint_id=previous.checkpoint.id,
                cause_id=cause.id,
            )
            self.save(None, None, None, cause)
        else:
            snapshot = self._snapshots.get(prefix)
            if snapshot is None:
                raise RuntimeError("FITD requested an unknown conversation prefix")
            change = self._restore("Target", snapshot.checkpoint, cause)
            self._history = prefix
            self._emit(
                "history_restored",
                role="Target",
                conversation_id=self._conversation_id,
                checkpoint_id=snapshot.checkpoint.id,
                from_checkpoint_id=previous.checkpoint.id,
                cause_id=cause.id,
            )
        current = self.current
        result = self._publish(current.result, change.context, current.checkpoint.id)
        return Synchronization(change, previous, current, result)

    def accept(self, cause: Cause) -> ResultChange:
        snapshot = self.current
        assert snapshot.response is not None
        self._snapshots[self._history] = replace(snapshot, result=snapshot.response)
        return self._publish(
            snapshot.response,
            replace(snapshot.checkpoint.context, cause=cause),
            snapshot.checkpoint.id,
        )

    def exchange(
        self, prompt: Conversation, request: Message, cause: Cause
    ) -> TargetExchange:
        synchronization = self.synchronize(prompt, cause)
        result = self.current.result
        turns = _history(prompt)
        exchange = self._send("Target", request, cause)
        response = exchange.response
        self._history = turns + (("assistant", response.to_string()),)
        point = self.save(request, response, result, cause)
        self._emit(
            "target_exchange",
            conversation_id=self._conversation_id,
            checkpoint_id=point.checkpoint.id,
            request_id=request.id,
            response_id=response.id,
        )
        return TargetExchange(exchange, synchronization, point)

    def generate(self, prompt: Conversation, generations_this_call: int):
        if generations_this_call != 1:
            raise ValueError("Only one generation is supported")
        assert self._request is not None
        outcome = self.exchange(
            prompt, self._request, Cause("target_exchange", self._attempt_id)
        )
        return [GarakMessage(outcome.exchange.response.to_string(), lang="en")]

    def clear_history(self):
        # garak clears its stateless generator after every exchange. LLMChat owns
        # the actual conversation and must retain it for the next call.
        pass


class _FITD(FITD):
    def __init__(
        self,
        base: Message,
        send: _Send,
        evaluate: _Evaluate,
        check_running: Callable[[], None],
        target: _Target,
        reset: _Reset,
        emit: _Emit,
        profile: _FITDProfile,
        conversation_id: _ConversationId,
    ) -> None:
        self._profile = profile
        self._base = base
        self._send = send
        self._evaluate = evaluate
        self._check_running = check_running
        self._target = target
        self._reset = reset
        self._emit = emit
        self._get_conversation_id = conversation_id
        self._attacker_conversation_id: str | None = None
        self._last_attacker_exchange: Exchange | None = None
        self._attempt_id: str | None = None
        self._decision_cause: Cause | None = None
        self._checks: list[LocalDecision] = []
        self._requests: dict[str, Message] = {}
        self.success = False
        self.failure: Exception | None = None
        super().__init__(config_root={})

    def raise_if_interrupted(self) -> None:
        if self.failure is not None:
            raise self.failure
        self._check_running()

    @contextmanager
    def _guard_execution(self) -> Iterator[None]:
        # Observe at model/evaluator boundaries as well as library branches.
        # Keep the first outcome even if garak catches or replaces an exception.
        try:
            self.raise_if_interrupted()
            yield
        except Exception as exc:
            if self.failure is None:
                self.failure = exc
            raise

    def _load_config(self, config_root):
        # Bind all execution settings locally instead of inheriting garak's
        # mutable global run settings or loading site/provider configuration.
        for key, value in self.DEFAULT_PARAMS.items():
            setattr(self, key, deepcopy(value))
        for key, value in asdict(self._profile).items():
            setattr(self, key, value)
        self.initial_prompt_count = 1
        self.follow_prompt_cap = False
        self.parallel_attempts = 1
        self.system_prompt = None

    def _get_langprovider(self):
        return _English()

    def _get_reverse_langprovider(self):
        return _English()

    def _buff_hook(self, attempts):
        return attempts

    def _get_red_team_response(self, prompt: str) -> str:
        # Includes FITD's internal alignment checks; no hidden model is loaded.
        with self._guard_execution():
            # The pinned implementation creates a fresh one-turn Conversation
            # for each red-team request. Preserve that semantics via LLMChat.
            cause = self._decision_cause or Cause("prepare_attack", self._attempt_id)
            self._reset("Attacker", cause)
            previous = self._attacker_conversation_id
            self._attacker_conversation_id = self._get_conversation_id("Attacker")
            request = self._base.spawn(prompt)
            self._emit(
                "history_reset",
                role="Attacker",
                conversation_id=self._attacker_conversation_id,
                previous_conversation_id=previous,
                request_id=request.id,
                cause_id=self._attempt_id,
            )
            exchange = self._send("Attacker", request, cause)
            self._last_attacker_exchange = exchange
            return exchange.response.to_string()

    def _is_refusal(self, response: str) -> bool:
        if self._attempt_id is None:
            exchange = self._last_attacker_exchange
            assert exchange is not None
            request, message, context = (
                exchange.request,
                exchange.response,
                exchange.context,
            )
            checkpoint_id = None
        else:
            point = self._target.current
            assert point.request is not None and point.response is not None
            assert self._decision_cause is not None
            request, message = point.request, point.response
            context = replace(point.checkpoint.context, cause=self._decision_cause)
            checkpoint_id = point.checkpoint.id
        decision = self.check_refusal(request, message, context, checkpoint_id)
        self._checks.append(decision)
        return decision.value

    def check_refusal(
        self,
        request: Message,
        response: Message,
        context: Context,
        checkpoint_id: str | None,
    ) -> LocalDecision:
        value = super()._is_refusal(response.to_string())
        decision = LocalDecision(
            "refusal", value, context, request, response, checkpoint_id
        )
        self._emit(
            "local_decision",
            decision="refusal",
            value=value,
            attempt_id=context.cause.attempt_id,
            response_id=response.id,
        )
        return decision

    def _is_response_aligned(self, request: str, response: str) -> bool:
        point = self._target.current
        assert point.request is not None and point.response is not None
        assert self._decision_cause is not None
        decision = self.check_alignment(
            point.request,
            point.response,
            replace(point.checkpoint.context, cause=self._decision_cause),
            point.checkpoint.id,
        )
        self._checks.append(decision)
        return decision.value

    def check_alignment(
        self,
        request: Message,
        response: Message,
        context: Context,
        checkpoint_id: str,
    ) -> LocalDecision:
        value = super()._is_response_aligned(request.to_string(), response.to_string())
        judgment = self._last_attacker_exchange
        assert judgment is not None
        decision = LocalDecision(
            "alignment", value, context, request, response, checkpoint_id, judgment
        )
        self._emit(
            "local_decision",
            decision="alignment",
            value=value,
            attempt_id=context.cause.attempt_id,
            response_id=response.id,
            judgment_response_id=judgment.response.id,
        )
        return decision

    def _create_init_attempts(self):
        with self._guard_execution():
            self.refusal_detector = Prefixes(config_root={})
            goal = self._base.to_string()
            self.target_prompts = [goal]
            initial = self._generate_init_turn(goal)
            if initial is None:
                return []
            self.level_queries_per_goal = {goal: self._get_level_queries(goal, initial)}
            attempt = self._create_attempt(initial)
            self._requests[str(attempt.uuid)] = self._base.spawn(initial)
            attempt.notes.update(
                {
                    "turn_num": 0,
                    "target_turn": goal,
                    "levels_completed": 0,
                    "is_adversarial": False,
                    "is_last_turn": False,
                    "last_non_refused_prompt": None,
                }
            )
            return [attempt]

    def _verify_is_last_response_success(self, attempt):
        # Native verification precedes refusal/alignment and can terminate on
        # material the adaptive branch would reject. Replace that unsafe-text
        # judge with our accepted-result evaluation *after* the native branch.
        # The library still performs every branch/check exactly once.
        return [False]

    def _generate_next_attempts(self, last_attempt):
        with self._guard_execution():
            self._attempt_id = str(last_attempt.uuid)
            try:
                return super()._generate_next_attempts(last_attempt)
            finally:
                self._attempt_id = None

    def generate_next_turn_attempt(self, conversation, last_attempt):
        continuation = self.select_continuation(conversation, last_attempt)
        decision, next_attempt = continuation.decision, continuation.attempt
        cause = replace(decision.context.cause, reason=decision.branch)
        if decision.branch == "continue":
            accepted = self._target.accept(cause)
            if last_attempt.notes["is_adversarial"]:
                assert accepted.current is not None
                evaluation = self._evaluate(
                    accepted.current, accepted.context, accepted.checkpoint_id
                )
                self.success = evaluation.evaluation.success
                if self.success:
                    return None
        if next_attempt is not None:
            # Even the final budgeted round applies the selected conversation.
            self._target.synchronize(next_attempt.prompt, cause)
        return next_attempt

    def select_continuation(self, conversation, last_attempt) -> _Continuation:
        # Observe the actual library-selected continuation; do not duplicate its
        # ladder indexing, refusal handling, rephrasing, or realignment logic.
        branch_id = str(uuid4())
        cause = Cause("select_continuation", str(last_attempt.uuid), branch_id)
        self._decision_cause = cause
        self._checks = []
        try:
            next_attempt = super().generate_next_turn_attempt(
                conversation, last_attempt
            )
        finally:
            self._decision_cause = None
        before = _history(conversation)
        branch: Literal["exhausted", "continue", "realign", "restart", "rephrase"]
        if next_attempt is None:
            branch = "exhausted"
        elif (
            next_attempt.notes["levels_completed"]
            > last_attempt.notes["levels_completed"]
        ):
            branch = "continue"
        else:
            prefix = _history(next_attempt.prompt)[:-1]
            if prefix == before:
                branch = "realign"
            elif not prefix:
                branch = "restart"
            elif prefix == before[:-2]:
                branch = "rephrase"
            else:
                raise RuntimeError("Unrecognized FITD conversation adaptation")
        point = self._target.current
        response = point.response
        assert response is not None
        next_request = None
        if next_attempt is not None:
            next_request = self._base.spawn(_history(next_attempt.prompt)[-1][1])
            self._requests[str(next_attempt.uuid)] = next_request
        decision = BranchDecision(
            branch_id,
            replace(point.checkpoint.context, cause=cause),
            branch,
            point,
            str(next_attempt.uuid) if next_attempt is not None else None,
            next_request,
            last_attempt.notes["levels_completed"],
            next_attempt.notes["levels_completed"]
            if next_attempt is not None
            else None,
            tuple(self._checks),
        )
        self._emit(
            "branch_selected",
            branch_id=branch_id,
            attempt_id=str(last_attempt.uuid),
            next_attempt_id=str(next_attempt.uuid)
            if next_attempt is not None
            else None,
            branch=branch,
            response_id=response.id,
            checkpoint_id=point.checkpoint.id,
            accepted=branch == "continue",
            levels_completed=last_attempt.notes["levels_completed"],
            next_levels_completed=next_attempt.notes["levels_completed"]
            if next_attempt is not None
            else None,
        )
        return _Continuation(decision, next_attempt)

    def _execute_attempt(self, attempt):
        with (
            self._guard_execution(),
            self._target.attempt(
                str(attempt.uuid), self._requests.pop(str(attempt.uuid))
            ),
        ):
            return super()._execute_attempt(attempt)

    def _execute_all(self, attempts):
        # The base implementation writes to a process-global reportfile. Keep
        # library execution/postprocessing, with reporting owned by our User.
        with self._guard_execution():
            return [
                self._postprocess_attempt(self._execute_attempt(attempt))
                for attempt in attempts
            ]


def run_fitd(
    base: Message,
    send: _Send,
    publish: _Publish,
    evaluate: _Evaluate,
    check_running: Callable[[], None],
    *,
    checkpoint: _Checkpoint,
    restore: _Restore,
    reset: _Reset,
    conversation_id: _ConversationId,
    emit: _Emit,
    profile: _FITDProfile = FITD_PROFILE,
) -> str:
    target = _Target(send, publish, checkpoint, restore, reset, emit, conversation_id)
    probe = _FITD(
        base,
        send,
        evaluate,
        check_running,
        target,
        reset,
        emit,
        profile,
        conversation_id,
    )
    probe.probe(target)
    # IterativeProbe catches GarakException. Never infer normal completion from
    # its returned attempts when an integration boundary recorded a failure.
    probe.raise_if_interrupted()
    if probe.success:
        return "evaluator_success"
    return "round_limit" if probe.attempt_queue else "strategy_exhausted"
