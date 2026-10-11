"""The text implementation retains immutable versions and provenance."""

import json
from dataclasses import FrozenInstanceError
from unittest import TestCase

import pytest

from core.primitives.messages.message.text_message import TextMessage


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
