"""Construct the public FITD User with its pinned native runtime."""

from collections.abc import Mapping
from dataclasses import replace
from typing import TypedDict, Unpack

from core.chats import LLMChat, ModelRole
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.messages import Message

from .contracts import StrategyRuntime
from .execution_state import ExecutionSettings
from .garak_adapter import GarakRuntime
from .profile import FITD_PROFILE
from .user import FITDUser as ExecutionUser


class UserOptions(TypedDict, total=False):
    max_calls_per_conv: int
    runtime: StrategyRuntime


def execution_settings(
    chats: dict[ModelRole, LLMChat], base: Message, setup: EvaluatorSetup, options: UserOptions,
) -> ExecutionSettings:
    if setup.chat is not None:
        chats["Evaluator"] = setup.chat
    if len({id(chat) for chat in chats.values()}) != len(chats):
        raise ValueError("Participants must use independent LLMChat instances")
    rounds = validate_rounds(options.get("max_calls_per_conv", FITD_PROFILE.max_calls_per_conv))
    return ExecutionSettings(
        chats, base, setup.evaluator, replace(FITD_PROFILE, max_calls_per_conv=rounds),
        options.get("runtime", GarakRuntime()),
    )


def validate_rounds(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("max_calls_per_conv must be a positive integer")
    return value


def validate_options(options: Mapping[str, object], allowed: frozenset[str]) -> None:
    if unknown := set(options) - allowed:
        raise TypeError(f"Unexpected FITD options: {', '.join(sorted(unknown))}")


class FITDUser(ExecutionUser):
    def __init__(
        self, target: LLMChat, attacker: LLMChat, base: Message,
        evaluator: Evaluator | EvaluatorSetup, **options: Unpack[UserOptions],
    ) -> None:
        validate_options(options, frozenset({"max_calls_per_conv", "runtime"}))
        super().__init__(execution_settings(
            {"Target": target, "Attacker": attacker}, base, evaluator_setup(evaluator), options,
        ))
