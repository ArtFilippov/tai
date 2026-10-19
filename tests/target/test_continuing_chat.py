"""Attacker checks retain their own complete conversation."""

import unittest

from garak.attempt import Conversation, Message, Turn

from target.continuing_chat import ContinuingChat
from target.scripted_chat import ScriptedChat


class TestContinuingChat(unittest.TestCase):
    """Verifies that successive prompts include earlier Attacker exchanges."""

    def test_history(self) -> None:
        chat = ScriptedChat(["First answer", "Second answer"])
        attacker = ContinuingChat(chat)
        attacker.reply(
            Conversation(
                [
                    Turn(
                        "user",
                        Message(text="First check")
                    )
                ]
            )
        )
        attacker.reply(
            Conversation(
                [
                    Turn(
                        "user",
                        Message(text="Second check")
                    )
                ]
            )
        )
        assert [turn.content.text for turn in chat.transcript(1).turns] == [
            "First check", "First answer", "Second check"
        ], "The Attacker's prior check was lost"
