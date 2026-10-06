"""Evaluator assembly, separate from the one-operation contract.

Source chats are explicit dependencies. Wrappers are installed at construction,
never by inspecting or changing an evaluator during a run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Self

from core.chats import LLMChat

from .configuration import EvaluatorSetup as SetupContract
from .configuration import evaluator_setup
from .evaluator import EvaluationChat, Evaluator
from .model_evaluator import (
    ANSWER_RUBRIC,
    ANSWER_RUBRIC_VERSION,
    MODEL_RUBRIC,
    MODEL_RUBRIC_VERSION,
    ModelEvaluator,
)
from .observation import ObservedChat
from .text_evaluator import ContainsTextEvaluator

__all__ = ["EvaluatorSetup", "evaluator_setup"]


class EvaluatorSetup(SetupContract):
    """Assembly metadata, never an additional interface required of evaluators.

    Plain evaluators need only ``evaluate``. For chat-based evaluators use
    ``with_chat`` to supply a factory, the source chat, and diagnostic settings.
    The model/contains_text constructors assemble the standard implementations.
    """

    @classmethod
    def with_chat(
        cls,
        chat: LLMChat,
        factory: Callable[[EvaluationChat], Evaluator],
        *,
        settings: Callable[[], Mapping[str, object]] | None = None,
    ) -> Self:
        return cls(factory(ObservedChat(chat)), chat, settings)

    @classmethod
    def model(
        cls,
        chat: LLMChat,
        rubric_version: str = MODEL_RUBRIC_VERSION,
        *,
        rubric: str = MODEL_RUBRIC,
    ) -> Self:
        return cls.with_chat(
            chat,
            lambda wrapped: ModelEvaluator(wrapped, rubric_version, rubric),
            settings=lambda: {
                "type": "model",
                "rubric_version": rubric_version,
                "rubric": rubric,
                "chat": chat.settings(),
            },
        )

    @classmethod
    def contains_text(cls, criterion: str) -> Self:
        return cls(
            ContainsTextEvaluator(criterion),
            configuration=lambda: {"type": "contains_text", "criterion": criterion},
        )

    @classmethod
    def answer(cls, chat: LLMChat) -> Self:
        return cls.model(chat, ANSWER_RUBRIC_VERSION, rubric=ANSWER_RUBRIC)
