"""Generation that can observe its owning exchange's cancellation signal."""

from threading import Event
from typing import Protocol

from .chat_turn_contract import ChatTurn


class EventGeneration(Protocol):
    """Complete one response while observing the supplied per-exchange signal."""

    def __call__(self, turns: tuple[ChatTurn, ...], cancellation: Event, /) -> str: ...
