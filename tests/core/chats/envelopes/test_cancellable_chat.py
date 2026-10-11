"""An interrupted exchange keeps the observed response out of its dialogue."""

from unittest import TestCase

from core.chats import ChatCancelled
from core.chats.envelopes.cancellable_chat import CancellableChat
from core.primitives.diagnostics.diagnostic_redactor.redaction import Redaction
from core.primitives.messages.message.text_message import TextMessage


class TestCancellableChat(TestCase):
    """Check that a per-exchange signal prevents committing an observed response."""

    def test_cancellation_retains_uncommitted_response(self) -> None:
        chat = CancellableChat(
            lambda turns, signal: (signal.set() or "observed"),
            model="scripted", redactor=Redaction(()),
        )
        with self.assertRaises(ChatCancelled) as failure:
            chat.send(TextMessage("request"))
        response = failure.exception.observed()
        outcome = (chat.history(), response.to_string()) if response is not None else None
        assert outcome == ((), "observed"), (
            "Cancellation committed the observed response or lost its evidence"
        )
