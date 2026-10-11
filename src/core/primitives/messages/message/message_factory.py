"""Construction seam for independent textual results."""

from typing import Protocol

from .message import Message


class MessageFactory(Protocol):
    """Create a fresh immutable Message for supplied text."""

    def __call__(self, text: str, /) -> Message: ...
