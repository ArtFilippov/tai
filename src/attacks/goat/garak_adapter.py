"""Pinned GOAT templates and O-T-S-R parser, with run-owned conversations."""

from copy import deepcopy
from dataclasses import dataclass
from typing import override

from garak.probes.goat import GOATAttack

from attacks.shared.contracts import GARAK_REVISION, CompletionReason, StrategyPorts, StrategyRuntime
from core.primitives.messages import Message
from core.runs.operations import Cause, Exchange, LocalDecision

from .profile import GOATProfile

UNSAFE = "unsafe"


class NativeGOAT(GOATAttack):
    """Load native resources without garak's dataset, globals or model plugin.

    Args:
        goal (str): The explicit base request used as the only native goal.
    """

    def __init__(self, goal: str) -> None:
        self._goal = goal
        super().__init__(config_root={})

    @override
    def _load_config(self, config_root: object = None) -> None:
        if config_root != {}:
            raise ValueError("GOAT requires run-local configuration")
        for key, value in self.DEFAULT_PARAMS.items():
            setattr(self, key, deepcopy(value))
        self.custom_goals = [self._goal]
        self.num_goals = 1
        self.generations = 1
        self.parallel_attempts = 1

    @override
    def _get_langprovider(self) -> None:
        return None

    @override
    def _get_reverse_langprovider(self) -> None:
        return None

    def parse(self, text: str) -> str | None:
        parsed = self._parse_otsr_response(self._extract_content(text))
        return parsed.get("Response") if parsed else None

    def is_unsafe(self, text: str) -> bool:
        return UNSAFE in self._extract_content(text).lower()

    def initial_prompt(self) -> str:
        goal = self._goal
        system = self.system_prompt_template.format(goal=goal, attack_techniques=self._format_attack_techniques())
        return f"{system}\n\n{self.initial_prompt_template.format(goal=goal)}"


class GOATProbe:
    def __init__(self, base: Message, ports: StrategyPorts, profile: GOATProfile) -> None:
        super().__init__()
        self.base, self.ports, self.profile = base, ports, profile
        self.native = NativeGOAT(base.to_string())

    def _judge_exchange(self, prompt: str, cause: Cause) -> Exchange:
        conversation = self.ports.conversation
        snapshot = conversation.checkpoint("Attacker", cause)
        self.ports.check_running()
        judgment = self.ports.send("Attacker", self.base.spawn(prompt), cause)
        self.ports.check_running()
        _ = conversation.restore("Attacker", snapshot, cause)
        return judgment

    def judge(self, exchange: Exchange, checkpoint_id: str, turn: int) -> LocalDecision:
        """Use garak's verify prompt and decision rule without losing attacker history.

        Args:
            exchange (Exchange): Completed Target exchange.
            checkpoint_id (str): Target history checkpoint.
            turn (int): Current GOAT turn.

        Returns:
            LocalDecision: Native judge decision and its independent evidence.
        """
        cause = Cause("goat_judge", str(turn))
        target_response = exchange.response
        prompt = self.native.judge_prompt_template.format(goal=self.base.to_string(), response=target_response.to_string())
        judgment = self._judge_exchange(prompt, cause)
        judge_response = judgment.response
        verdict = self.native.is_unsafe(judge_response.to_string())
        return LocalDecision("goat_judge", verdict, exchange.context, exchange.request, target_response, checkpoint_id, judgment)

    def _prompt(self, previous: tuple[Message, Message] | None) -> str:
        goal, native = self.base.to_string(), self.native
        if previous is not None:
            request, response = previous
            return native.followup_prompt_template.format(
                goal=goal, adversarial_prev_prompt=request.to_string(), prev_model_response=response.to_string(),
            )
        return native.initial_prompt()

    def _turn(self, turn: int, previous: tuple[Message, Message] | None) -> tuple[CompletionReason | None, tuple[Message, Message] | None]:
        cause = Cause("goat_turn", str(turn))
        attacker = self.ports.send("Attacker", self.base.spawn(self._prompt(previous)), cause)
        if not (text := self._parse_attacker(attacker)):
            return "strategy_exhausted", previous
        request = self.base.spawn(text)
        target = self.ports.send("Target", request, cause)
        checkpoint_id = self._present(target, cause)
        verdict = self.judge(target, checkpoint_id, turn)
        if self._assess(target, checkpoint_id):
            return "evaluator_success", None
        if verdict.value:
            return "judge_stop", None
        return None, (request, target.response)

    def _parse_attacker(self, attacker: Exchange) -> str | None:
        response = attacker.response
        return self.native.parse(response.to_string())

    def _assess(self, target: Exchange, checkpoint_id: str) -> bool:
        assessment = self.ports.evaluate(target.response, target.context, checkpoint_id)
        return assessment.evaluation.success

    def _present(self, exchange: Exchange, cause: Cause) -> str:
        conversation = self.ports.conversation
        checkpoint = conversation.checkpoint("Target", cause)
        _ = self.ports.publish(exchange.response, exchange.context, checkpoint.id)
        return checkpoint.id

    def __call__(self) -> CompletionReason:
        previous: tuple[Message, Message] | None = None
        for turn in range(1, self.profile.max_calls_per_conv + 1):
            reason, previous = self._turn(turn, previous)
            if reason is not None:
                return reason
        return "round_limit"


@dataclass(frozen=True)
class GOATRuntime(StrategyRuntime):
    profile: GOATProfile
    probe_type: type[GOATProbe] = GOATProbe

    @property
    @override
    def revision(self) -> str:
        return GARAK_REVISION

    @override
    def __call__(self, base: Message, ports: StrategyPorts) -> CompletionReason:
        return self.probe_type(base, ports, self.profile)()
