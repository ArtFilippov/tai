"""Construct the public FITD User with its pinned native runtime."""

from dataclasses import replace
from typing import TypedDict, Unpack

from core.chats import LLMChat, ModelRole
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.messages import Message
from attacks.shared.assembly import participants, validate_options
from attacks.shared.contracts import StrategyRuntime
from attacks.shared.execution_state import ExecutionSettings
from attacks.shared.user import ExecutionUser

from .garak_adapter import GarakRuntime
from .profile import FITD_PROFILE


class UserOptions(TypedDict, total=False):
    max_calls_per_conv: int
    runtime: StrategyRuntime


def execution_settings(
    chats: dict[ModelRole, LLMChat], base: Message, setup: EvaluatorSetup, options: UserOptions,
) -> ExecutionSettings:
    rounds = validate_rounds(options.get("max_calls_per_conv", FITD_PROFILE.max_calls_per_conv))
    profile = replace(FITD_PROFILE, max_calls_per_conv=rounds)
    return ExecutionSettings(
        participants(chats, setup), base, setup.evaluator, profile.configuration(),
        options.get("runtime", GarakRuntime(profile=profile)),
    )


def validate_rounds(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("max_calls_per_conv must be a positive integer")
    return value


class FITDUser(ExecutionUser):
    def __init__(
        self, target: LLMChat, attacker: LLMChat, base: Message,
        evaluator: Evaluator | EvaluatorSetup, **options: Unpack[UserOptions],
    ) -> None:
        validate_options(options, frozenset({"max_calls_per_conv", "runtime"}))
        super().__init__(execution_settings(
            {"Target": target, "Attacker": attacker}, base, evaluator_setup(evaluator), options,
        ))
