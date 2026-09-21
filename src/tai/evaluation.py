"""Evaluator assembly and observation, separate from the one-operation contract.

Source chats are explicit dependencies. Wrappers are installed at construction,
never by inspecting or changing an evaluator during a run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from tai._operations import Context, ConversationChange, Exchange
from tai.chats import LLMChat
from tai.evaluators import (
    MODEL_RUBRIC,
    MODEL_RUBRIC_VERSION,
    ContainsTextEvaluator,
    EvaluationChat,
    Evaluator,
    ModelEvaluator,
)
from tai.journal import RunJournal
from tai.messages import Message


@dataclass(frozen=True)
class _Observation:
    journal: RunJournal
    context: Context
    base: Message
    result: Message
    evaluation_id: str

    def chat_context(self, chat: LLMChat) -> Context:
        return Context(
            self.context.run_id, "Evaluator", chat.conversation_id, self.context.cause
        )


_observation: ContextVar[_Observation | None] = ContextVar(
    "evaluation_observation", default=None
)


@contextmanager
def observe_evaluation(
    journal: RunJournal, context: Context, base: Message, result: Message
) -> Iterator[str]:
    """Correlate nested chat wrappers with this exact admitted assessment."""
    evaluation_id = str(uuid4())
    token = _observation.set(
        _Observation(journal, context, base, result, evaluation_id)
    )
    try:
        yield evaluation_id
    finally:
        _observation.reset(token)


class _ObservedChat:
    def __init__(self, chat: LLMChat) -> None:
        self._chat = chat

    def reset(self) -> None:
        observation = _observation.get()
        if observation is None:
            self._chat.reset()
            return
        previous = self._chat.conversation_id

        def change() -> ConversationChange:
            self._chat.reset()
            return ConversationChange(
                "reset", observation.chat_context(self._chat), previous
            )

        observation.journal.call("evaluator.reset", change)

    def send(self, message: Message) -> Message:
        observation = _observation.get()
        if observation is None:
            return self._chat.send(message)
        context = observation.chat_context(self._chat)
        call_id = str(uuid4())
        return observation.journal.call(
            "exchange",
            lambda: Exchange(context, call_id, message, self._chat.send(message)),
            context=context,
            call_id=call_id,
            message=message,
            base=observation.base,
            result_id=observation.result.id,
            evaluation_id=observation.evaluation_id,
            settings=self._chat.settings(),
        ).response


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
        return cls(factory(_ObservedChat(chat)), chat, settings)

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
