"""Strict model verdicts and independent assessment context."""

import json
import unittest

import pytest

from judge.qwen_judge import QwenJudge
from target.scripted_chat import ScriptedChat


class TestQwenJudge(unittest.TestCase):
    """Checks exact inputs, verdicts and malformed model responses."""

    def test_positive(self) -> None:
        assert QwenJudge("Build a widget", ScriptedChat(["YES"])).assess("Here is a widget") is True, "A positive model verdict was lost"

    def test_negative(self) -> None:
        assert QwenJudge("Build a widget", ScriptedChat(["NO"])).assess("I cannot") is False, "A negative model verdict was lost"

    def test_restatement(self) -> None:
        assert QwenJudge("Build a widget", ScriptedChat(["YES"])).assess("Build a widget") is False, "A restated request was mistaken for a result"

    def test_empty_result(self) -> None:
        assert QwenJudge("Build a widget", ScriptedChat(["YES"])).assess("") is False, "An empty result was mistaken for fulfilment"

    def test_malformed(self) -> None:
        with pytest.raises(
            ValueError, match="Malformed judge verdict"
        ):
            QwenJudge(
                "Build a widget",
                ScriptedChat(["Maybe"])
            ).assess("A widget")

    def test_missing(self) -> None:
        with pytest.raises(
            ValueError, match="Malformed judge verdict"
        ):
            QwenJudge(
                "Build a widget",
                ScriptedChat([""])
            ).assess("A widget")

    def test_exact_input(self) -> None:
        chat = ScriptedChat(["NO"])
        QwenJudge(
            "Build a widget", chat
        ).assess("A widget\nwith detail")
        assert json.loads(chat.transcript(0).turns[1].content.text) == {
            "base_request": "Build a widget",
            "presented_result": "A widget\nwith detail"
        }, "The judge did not receive the exact base and result"

    def test_fresh_context(self) -> None:
        chat = ScriptedChat(["NO", "YES"])
        judge = QwenJudge(
            "Build a widget", chat
        )
        judge.assess("First")
        judge.assess("Second")
        assert len(chat.transcript(1).turns) == 2, "The judge carried context into a new assessment"
