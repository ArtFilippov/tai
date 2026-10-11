"""The immutable textual input and result contract."""

from .message.message import Message
from .message.message_factory import MessageFactory

__all__ = ["Message", "MessageFactory"]
