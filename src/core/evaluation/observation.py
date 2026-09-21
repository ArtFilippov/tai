"""Correlate evaluator chat exchanges with the admitted result assessment."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import override
from uuid import uuid4

from core.chats import LLMChat
from core.primitives.messages import Message
from core.runs.journal_ports import Journal
from core.runs.operations import Context, ConversationChange, Exchange

from .evaluator import EvaluationChat


@dataclass(frozen=True)
class _Observation:
    journal: Journal
    context: Context
    base: Message
    result: Message
    evaluation_id: str

    def _chat_context(self, chat: LLMChat) -> Context:
        return Context(
            self.context.run_id, "Evaluator", chat.conversation_id, self.context.cause
        )

    def reset(self, chat: LLMChat) -> None:
        previous = chat.conversation_id

        def change() -> ConversationChange:
            chat.reset()
            return ConversationChange("reset", self._chat_context(chat), previous)

        _ = self.journal.call("evaluator.reset", change)

    def send(self, chat: LLMChat, message: Message) -> Message:
        context = self._chat_context(chat)
        call_id = str(uuid4())
        return self.journal.call(
            "exchange",
            lambda: Exchange(context, call_id, message, chat.send(message)),
            context=context,
            call_id=call_id,
            message=message,
            base=self.base,
            result_id=self.result.id,
            evaluation_id=self.evaluation_id,
            settings=chat.settings(),
        ).response


_observation: ContextVar[_Observation | None] = ContextVar(
    "evaluation_observation", default=None
)


@contextmanager
def observe_evaluation(
    journal: Journal, context: Context, base: Message, result: Message
) -> Generator[str]:
    """Correlate nested chat wrappers with this exact admitted assessment.

    Args:
        journal (Journal): Receiving journal.
        context (Context): Admitted assessment context.
        base (Message): Explicit original request.
        result (Message): Complete presented result.

    Yields:
        str: Unique identifier shared by the assessment's nested exchanges.
    """
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
        super().__init__()
        self._chat = chat

    @override
    def reset(self) -> None:
        if (observation := _observation.get()) is None:
            self._chat.reset()
            return
        observation.reset(self._chat)

    @override
    def send(self, message: Message) -> Message:
        if (observation := _observation.get()) is None:
            return self._chat.send(message)
        return observation.send(self._chat, message)
