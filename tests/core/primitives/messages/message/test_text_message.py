"""The text implementation retains immutable versions and provenance."""

import json
from dataclasses import FrozenInstanceError
from unittest import TestCase

import pytest

from core.chats.envelopes.openai import OpenAIChat
from core.primitives.diagnostics.diagnostic_redactor.redaction import Redaction
from core.primitives.messages.message.text_message import TextMessage
from core.runs.journal import RunJournal


class TestTextMessage(TestCase):
    """Verify that versions retain their own text and origin."""

    def message(self) -> TextMessage:
        return TextMessage("request")

    def test_transform(self) -> None:
        base = self.message()
        changed = base.transform(str.upper)
        assert (changed.to_string(), changed.base(), changed.revert(1) is base) == (
            "REQUEST", "request", True,
        ), "Transformation lost its original text"

    def test_spawn(self) -> None:
        base = self.message().transform(str.upper)
        result = base.spawn("reply")
        assert (result.base(), result.origin is base) == (
            "reply", True,
        ), "Spawn did not keep an independent base and the exact origin"

    def test_snapshot(self) -> None:
        result = self.message().transform(str.upper).spawn("reply")
        assert (json.loads(result.dump())["origin"]["text"], result.snapshot()["base"]) == (
            "REQUEST", "reply",
        ), "Serialization lost the computed history"

    def test_immutable(self) -> None:
        assert pytest.raises(FrozenInstanceError, setattr, self.message(), "text", "other").value, (
            "Text result was mutable"
        )

    def test_journal_redacts_escaped_message_history(self) -> None:
        journal = RunJournal(redactor=Redaction(("café",)))
        journal.bind("run")
        journal.record("exchange", response=self.message().spawn("café").spawn('"caf\\u00e9"'))
        assert "\\u00e9" not in journal.events()[-1].dump(), (
            "Journal exposed an escaped credential in a Message"
        )

    def test_chat_settings_redact_escaped_secret_and_preserve_options(self) -> None:
        chat = OpenAIChat(
            base_url="https://example.test/v1", model="model", initial=self.message().spawn("café"),
            api_key="key-123", generation={"note": "caf\\u00e9", "temperature": 0.5},
            redaction=Redaction,
        )
        redactor = Redaction(("café",)).combine(chat.redactor)
        settings = json.dumps(chat.diagnostic_settings(redactor), ensure_ascii=False)
        assert all((
            "café" not in settings, "caf\\u00e9" not in settings,
            "key-123" not in settings, '"temperature": 0.5' in settings,
        )), (
            "Chat settings exposed a secret or lost ordinary options"
        )
