"""The owner of completed turns and opaque, conversation-local snapshots."""

from typing import Protocol

from core.primitives.messages import Message

from .chat_turn_contract import ChatTurn


class ConversationHistory(Protocol):
    """Own the current branch, initial instructions and saved branches."""

    def history(self) -> tuple[ChatTurn, ...]: ...

    def identity(self) -> str: ...

    def checkpoint(self) -> object: ...

    def accept(self, turns: tuple[ChatTurn, ...], response: Message) -> None: ...

    def restore(self, token: object) -> None: ...

    def reset(self) -> None: ...
