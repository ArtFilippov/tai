"""The public Message interface identifies a textual result."""

from unittest import TestCase

from core.primitives.messages import Message
from core.primitives.messages.message.text_message import TextMessage


class TestMessage(TestCase):
    """Verify the public contract of a text result."""

    def test_contract(self) -> None:
        assert isinstance(TextMessage("request"), Message), "Text result did not satisfy Message"
