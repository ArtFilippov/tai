"""Pinned garak integration. No library types cross this module's boundary.

FITD retains its initial-turn builder, ladder builder, adaptive branch logic,
and IterativeProbe loop. Overrides bind task, model I/O, evaluation, and reporting
to this run. No global configuration or plugin registry is modified.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from uuid import uuid4

from garak.attempt import Conversation
from garak.attempt import Message as GarakMessage
from garak.detectors.mitigation import Prefixes
from garak.probes.fitd import FITD

from tai._profile import FITD_PROFILE, _FITDProfile
from tai.chats import ModelRole
from tai.messages import Message

GARAK_REVISION = "3f50ea5ff9cd7050099940647c15c39b07a93392"

type _Send = Callable[[ModelRole, Message], Message]
type _Checkpoint = Callable[[ModelRole], object]
type _Restore = Callable[[ModelRole, object], None]
type _Reset = Callable[[ModelRole], None]
type _Emit = Callable[..., None]
type _History = tuple[tuple[str, str], ...]


class _English:
    """The supported English profile needs no translation service or models."""

    target_lang = "en"

    def get_text(self, prompts):
        return prompts


@dataclass(frozen=True)
class _Snapshot:
    token: object
    id: str
    response: Message | None
    result: Message | None


def _history(conversation: Conversation) -> _History:
    return tuple((turn.role, turn.content.text) for turn in conversation.turns)


class _Target:
    """Map library prefixes to chat-owned snapshots, never provider histories.

    A completed response is diagnostic until FITD chooses to advance its ladder.
    Realignment retains off-topic context without presenting it as a result.
    """

    def __init__(
        self,
        base: Message,
        send: _Send,
        publish: Callable[[Message | None], None],
        checkpoint: _Checkpoint,
        restore: _Restore,
        reset: _Reset,
        emit: _Emit,
    ) -> None:
        self._base = base
        self._send = send
        self._publish = publish
        self._checkpoint = checkpoint
        self._restore = restore
        self._reset = reset
        self._emit = emit
        self._history: _History = ()
        self._snapshots: dict[_History, _Snapshot] = {}
        self._conversation_id = str(uuid4())
        self._save(None, None)

    @property
    def current(self) -> _Snapshot:
        return self._snapshots[self._history]

    def _save(self, response: Message | None, result: Message | None) -> None:
        snapshot = _Snapshot(self._checkpoint("Target"), str(uuid4()), response, result)
        self._snapshots[self._history] = snapshot
        self._emit(
            "checkpoint_created",
            role="Target",
            conversation_id=self._conversation_id,
            checkpoint_id=snapshot.id,
            response_id=response.id if response is not None else None,
            result_id=result.id if result is not None else None,
        )

    def synchronize(self, prompt: Conversation, *, cause_id: str | None = None) -> None:
        turns = _history(prompt)
        if not turns or turns[-1][0] != "user":
            raise RuntimeError("FITD must supply a final user turn")
        prefix = turns[:-1]
        if prefix == self._history:
            return
        previous = self.current
        if not prefix:
            self._reset("Target")
            old_conversation_id = self._conversation_id
            self._conversation_id = str(uuid4())
            self._history = ()
            self._snapshots.clear()
            self._emit(
                "history_reset",
                role="Target",
                conversation_id=self._conversation_id,
                previous_conversation_id=old_conversation_id,
                from_checkpoint_id=previous.id,
                cause_id=cause_id,
            )
            self._save(None, None)
        else:
            snapshot = self._snapshots.get(prefix)
            if snapshot is None:
                raise RuntimeError("FITD requested an unknown conversation prefix")
            self._restore("Target", snapshot.token)
            self._history = prefix
            self._emit(
                "history_restored",
                role="Target",
                conversation_id=self._conversation_id,
                checkpoint_id=snapshot.id,
                from_checkpoint_id=previous.id,
                cause_id=cause_id,
            )
        self._publish(self.current.result)

    def accept(self) -> None:
        snapshot = self.current
        assert snapshot.response is not None
        self._snapshots[self._history] = replace(snapshot, result=snapshot.response)
        self._publish(snapshot.response)

    def generate(self, prompt: Conversation, generations_this_call: int):
        if generations_this_call != 1:
            raise ValueError("Only one generation is supported")
        self.synchronize(prompt)
        result = self.current.result
        turns = _history(prompt)
        request = self._base.spawn(turns[-1][1])
        response = self._send("Target", request)
        self._history = turns + (("assistant", response.to_string()),)
        self._save(response, result)
        self._emit(
            "target_exchange",
            conversation_id=self._conversation_id,
            checkpoint_id=self.current.id,
            request_id=request.id,
            response_id=response.id,
        )
        return [GarakMessage(response.to_string(), lang="en")]

    def clear_history(self):
        # garak clears its stateless generator after every exchange. LLMChat owns
        # the actual conversation and must retain it for the next call.
        pass


class _FITD(FITD):
    def __init__(
        self,
        base: Message,
        send: _Send,
        evaluate: Callable[[], bool],
        check_running: Callable[[], None],
        target: _Target,
        reset: _Reset,
        emit: _Emit,
        profile: _FITDProfile,
    ) -> None:
        self._profile = profile
        self._base = base
        self._send = send
        self._evaluate = evaluate
        self._check_running = check_running
        self._target = target
        self._reset = reset
        self._emit = emit
        self._attacker_conversation_id: str | None = None
        self._last_attacker_response_id: str | None = None
        self._attempt_id: str | None = None
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
            self._reset("Attacker")
            previous = self._attacker_conversation_id
            self._attacker_conversation_id = str(uuid4())
            request = self._base.spawn(prompt)
            self._emit(
                "history_reset",
                role="Attacker",
                conversation_id=self._attacker_conversation_id,
                previous_conversation_id=previous,
                request_id=request.id,
                cause_id=self._attempt_id,
            )
            response = self._send("Attacker", request)
            self._last_attacker_response_id = response.id
            return response.to_string()

    def _is_refusal(self, response: str) -> bool:
        value = super()._is_refusal(response)
        current = self._target.current.response
        self._emit(
            "local_decision",
            decision="refusal",
            value=value,
            attempt_id=self._attempt_id,
            response_id=current.id
            if self._attempt_id is not None and current is not None
            else self._last_attacker_response_id,
        )
        return value

    def _is_response_aligned(self, request: str, response: str) -> bool:
        value = super()._is_response_aligned(request, response)
        current = self._target.current.response
        self._emit(
            "local_decision",
            decision="alignment",
            value=value,
            attempt_id=self._attempt_id,
            response_id=current.id if current is not None else None,
            judgment_response_id=self._last_attacker_response_id,
        )
        return value

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
        # Observe the actual library-selected continuation; do not duplicate its
        # ladder indexing, refusal handling, rephrasing, or realignment logic.
        next_attempt = super().generate_next_turn_attempt(conversation, last_attempt)
        before = _history(conversation)
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
        response = self._target.current.response
        assert response is not None
        branch_id = str(uuid4())
        self._emit(
            "branch_selected",
            branch_id=branch_id,
            attempt_id=str(last_attempt.uuid),
            next_attempt_id=str(next_attempt.uuid)
            if next_attempt is not None
            else None,
            branch=branch,
            response_id=response.id,
            checkpoint_id=self._target.current.id,
            accepted=branch == "continue",
            levels_completed=last_attempt.notes["levels_completed"],
            next_levels_completed=next_attempt.notes["levels_completed"]
            if next_attempt is not None
            else None,
        )
        if branch == "continue":
            self._target.accept()
            if last_attempt.notes["is_adversarial"]:
                self.success = self._evaluate()
                if self.success:
                    return None
        if next_attempt is not None:
            # Apply the selected history even on the last budgeted round, when
            # IterativeProbe prepares a continuation but never sends it.
            self._target.synchronize(next_attempt.prompt, cause_id=branch_id)
        return next_attempt

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
    publish: Callable[[Message | None], None],
    evaluate: Callable[[], bool],
    check_running: Callable[[], None],
    *,
    checkpoint: _Checkpoint,
    restore: _Restore,
    reset: _Reset,
    emit: _Emit,
    profile: _FITDProfile = FITD_PROFILE,
) -> str:
    target = _Target(base, send, publish, checkpoint, restore, reset, emit)
    probe = _FITD(base, send, evaluate, check_running, target, reset, emit, profile)
    probe.probe(target)
    # IterativeProbe catches GarakException. Never infer normal completion from
    # its returned attempts when an integration boundary recorded a failure.
    probe.raise_if_interrupted()
    if probe.success:
        return "evaluator_success"
    return "round_limit" if probe.attempt_queue else "strategy_exhausted"
