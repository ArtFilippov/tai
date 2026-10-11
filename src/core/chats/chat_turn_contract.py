"""A completed conversation turn exposes its role and immutable Message."""

from typing import Protocol

from core.primitives.messages import Message


class ChatTurn(Protocol):
    """A role-labelled Message in one LLMChat's completed dialogue."""

    @property
    def role(self) -> str: ...

    @property
    def message(self) -> Message: ...

    def text(self) -> str: ...

    def dump(self) -> str: ...
