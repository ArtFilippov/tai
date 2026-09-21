"""Correlate evaluator chat exchanges with the admitted result assessment."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import override
from uuid import uuid4

from core.chats import LLMChat
from core.primitives.messages import Message
from core.runs.journal import RunJournal
from core.runs.operations import Context, ConversationChange, Exchange

from .evaluator import EvaluationChat


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


class ObservedChat(EvaluationChat):
    def __init__(self, chat: LLMChat) -> None:
        self._chat = chat

    @override
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

    @override
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
