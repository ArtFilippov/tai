"""Evaluator assembly, separate from the one-operation contract.

Source chats are explicit dependencies. Wrappers are installed at construction,
never by inspecting or changing an evaluator during a run.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from core.chats import LLMChat

from .evaluator import EvaluationChat, Evaluator
from .model_evaluator import (
    MODEL_RUBRIC,
    MODEL_RUBRIC_VERSION,
    ModelEvaluator,
)
from .observation import ObservedChat
from .text_evaluator import ContainsTextEvaluator


@dataclass(frozen=True)
class EvaluatorSetup:
    """Assembly metadata, never an additional interface required of evaluators.

    Plain evaluators need only ``evaluate``. For chat-based evaluators use
    ``with_chat`` to supply a factory, the source chat, and diagnostic settings.
    The model/contains_text constructors assemble the standard implementations.
    """

    evaluator: Evaluator
    chat: LLMChat | None = None
    configuration: Callable[[], dict[str, Any]] | None = None

    def settings(self) -> dict[str, Any]:
        """Detached configuration only; observation never invokes the evaluator."""
        if self.configuration is not None:
            return deepcopy(self.configuration())
        cls = type(self.evaluator)
        return {"type": f"{cls.__module__}.{cls.__qualname__}"}

    @classmethod
    def with_chat(
        cls,
        chat: LLMChat,
        factory: Callable[[EvaluationChat], Evaluator],
        *,
        settings: Callable[[], dict[str, Any]] | None = None,
    ) -> EvaluatorSetup:
        return cls(factory(ObservedChat(chat)), chat, settings)

    @classmethod
    def model(
        cls,
        chat: LLMChat,
        rubric_version: str = MODEL_RUBRIC_VERSION,
        *,
        rubric: str = MODEL_RUBRIC,
    ) -> EvaluatorSetup:
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
    def contains_text(cls, criterion: str) -> EvaluatorSetup:
        return cls(
            ContainsTextEvaluator(criterion),
            configuration=lambda: {"type": "contains_text", "criterion": criterion},
        )


def evaluator_setup(evaluator: Evaluator | EvaluatorSetup) -> EvaluatorSetup:
    return (
        evaluator
        if isinstance(evaluator, EvaluatorSetup)
        else EvaluatorSetup(evaluator)
    )
