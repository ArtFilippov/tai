"""Mutable branch state belonging to one LLMChat."""

from typing import override
from uuid import uuid4

from core.primitives.messages import Message

from ..chat_turn_contract import ChatTurn
from ..conversation_history import ConversationHistory
from .turn import Turn


class ConversationState(ConversationHistory):
    """Retain completed turns and identity-based, conversation-local snapshots."""

    def __init__(self, initial: Message | None) -> None:
        """Start with the supplied system Message.

        Args:
            initial (Message | None): Prepared system instructions, if any.
        """
        super().__init__()
        system: tuple[ChatTurn, ...] = (Turn(("system", initial)),) if initial is not None else ()
        self.turns: list[ChatTurn] = list(system)
        self.system: list[ChatTurn] = list(system)
        self.key: str = str(uuid4())
        self.checkpoints: dict[object, list[ChatTurn]] = {}

    @override
    def history(self) -> tuple[ChatTurn, ...]:
        return tuple(self.turns)

    @override
    def identity(self) -> str:
        return self.key

    @override
    def checkpoint(self) -> object:
        token = object()
        self.checkpoints[token] = self.turns.copy()
        return token

    @override
    def accept(self, turns: tuple[ChatTurn, ...], response: Message) -> None:
        self.turns[:] = (*turns, Turn(("assistant", response)))

    @override
    def reset(self) -> None:
        self.key = str(uuid4())
        self.turns[:] = self.system
        self.checkpoints.clear()

    @override
    def restore(self, token: object) -> None:
        # Never invoke caller-defined equality or hashing at this opaque boundary.
        for known, turns in self.checkpoints.items():
            if known is token:
                self.turns[:] = turns
                return
        raise ValueError("Foreign or stale LLMChat checkpoint")
