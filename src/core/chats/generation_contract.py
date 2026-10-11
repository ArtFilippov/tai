"""One text completion over an immutable request dialogue."""

from typing import Protocol

from .chat_turn_contract import ChatTurn


class Generation(Protocol):
    """Produce the next full model response from completed turns."""

    def __call__(self, turns: tuple[ChatTurn, ...], /) -> str: ...
