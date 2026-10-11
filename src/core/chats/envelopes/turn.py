"""One immutable turn in a model conversation."""

from dataclasses import dataclass
from typing import override

from core.primitives.messages import Message

from ..chat_turn_contract import ChatTurn


@dataclass(frozen=True, slots=True)
class Turn(ChatTurn):
    """Keep a completed Message paired with the speaker's role.

    Attributes:
        pair (tuple[str, Message]): Speaker and its immutable content.
    """

    pair: tuple[str, Message]

    @property
    @override
    def role(self) -> str:
        return self.pair[0]

    @property
    @override
    def message(self) -> Message:
        return self.pair[1]

    @override
    def text(self) -> str:
        return self.pair[1].to_string()

    @override
    def dump(self) -> str:
        return self.pair[1].dump()
