"""Pinned garak integration. No library types cross this module's boundary.

FITD retains its initial-turn builder, ladder builder, adaptive branch logic,
and IterativeProbe loop. Overrides bind task, model I/O, evaluation, and reporting
to this run. No global configuration or plugin registry is modified.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict

from garak.attempt import Conversation
from garak.attempt import Message as GarakMessage
from garak.detectors.mitigation import Prefixes
from garak.probes.fitd import FITD

from tai._profile import FITD_PROFILE
from tai.chats import ModelRole
from tai.messages import Message

GARAK_REVISION = "3f50ea5ff9cd7050099940647c15c39b07a93392"

type _Send = Callable[[ModelRole, Message], Message]


class _English:
    """The supported English profile needs no translation service or models."""

    target_lang = "en"

    def get_text(self, prompts):
        return prompts


class _Target:
    def __init__(self, send: _Send, publish: Callable[[Message], None]) -> None:
        self._send = send
        self._publish = publish
        self._history: list[tuple[str, str]] = []

    def generate(self, prompt: Conversation, generations_this_call: int):
        if generations_this_call != 1:
            raise ValueError("Only one generation is supported")
        turns = [(turn.role, turn.content.text) for turn in prompt.turns]
        if turns[:-1] != self._history or turns[-1][0] != "user":
            raise RuntimeError("FITD conversation requires a history restore")
        response = self._send("Target", Message(turns[-1][1]))
        self._history = turns + [("assistant", response.to_string())]
        self._publish(response)
        return [GarakMessage(response.to_string(), lang="en")]

    def clear_history(self):
        # garak clears its stateless generator after every exchange. LLMChat owns
        # the actual conversation and must retain it for the next call.
        pass


class _FITD(FITD):
    def __init__(
        self, base: Message, send: _Send, evaluate: Callable[[], bool]
    ) -> None:
        self._base = base
        self._send = send
        self._evaluate = evaluate
        self.success = False
        self.failure: Exception | None = None
        super().__init__(config_root={})

    def _load_config(self, config_root):
        # Bind all execution settings locally instead of inheriting garak's
        # mutable global run settings or loading site/provider configuration.
        for key, value in self.DEFAULT_PARAMS.items():
            setattr(self, key, deepcopy(value))
        for key, value in asdict(FITD_PROFILE).items():
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
        return self._send("Attacker", Message(prompt)).to_string()

    def _create_init_attempts(self):
        try:
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
        except Exception as exc:
            self.failure = exc
            raise

    def _verify_is_last_response_success(self, attempt):
        if not attempt.notes["is_adversarial"]:
            return [False]
        self.success = self._evaluate()
        return [self.success]

    def _execute_all(self, attempts):
        # The base implementation writes to a process-global reportfile. Keep
        # library execution/postprocessing, with reporting owned by our User.
        try:
            return [
                self._postprocess_attempt(self._execute_attempt(attempt))
                for attempt in attempts
            ]
        except Exception as exc:
            self.failure = exc
            raise


def run_fitd(
    base: Message,
    send: _Send,
    publish: Callable[[Message], None],
    evaluate: Callable[[], bool],
) -> str:
    probe = _FITD(base, send, evaluate)
    probe.probe(_Target(send, publish))
    # IterativeProbe catches GarakException. Never infer normal completion from
    # its returned attempts when an integration boundary recorded a failure.
    if probe.failure is not None:
        raise probe.failure
    if probe.success:
        return "evaluator_success"
    return "round_limit" if probe.attempt_queue else "strategy_exhausted"
