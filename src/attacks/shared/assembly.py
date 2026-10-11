"""Participant and option validation shared by strategy composition roots."""

from collections.abc import Mapping
from typing import Literal

from core.chats import LLMChat
from core.evaluation.configuration import EvaluatorSetup

type ModelRole = Literal["Target", "Attacker", "Evaluator"]


def participants(chats: dict[ModelRole, LLMChat], setup: EvaluatorSetup) -> dict[ModelRole, LLMChat]:
    if setup.chat is not None:
        chats["Evaluator"] = setup.chat
    if len({id(chat) for chat in chats.values()}) != len(chats):
        raise ValueError("Participants must use independent LLMChat instances")
    return chats


def validate_options(options: Mapping[str, object], allowed: frozenset[str]) -> None:
    if unknown := set(options) - allowed:
        raise TypeError(f"Unexpected attack options: {', '.join(sorted(unknown))}")
