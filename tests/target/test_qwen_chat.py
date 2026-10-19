"""Shared weights receive only the supplied conversation."""

import unittest

from garak.attempt import Conversation, Message, Turn

from target.qwen_chat import QwenChat
from target.scripted_model import ScriptedModel


class TestQwenChat(unittest.TestCase):
    """Checks that histories do not leak between model roles."""

    def test_role_isolation(self) -> None:
        model = ScriptedModel(
            [
                [
                    {
                        "role": "user",
                        "content": "Target secret"
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Attacker task"
                    }
                ]
            ]
        )
        target = QwenChat(model)
        attacker = QwenChat(model)
        target.reply(
            Conversation(
                [
                    Turn(
                        "user",
                        Message(text="Target secret")
                    )
                ]
            )
        )
        response = attacker.reply(
            Conversation(
                [
                    Turn(
                        "user",
                        Message(text="Attacker task")
                    )
                ]
            )
        )
        assert response == "Reply", "The Target history leaked into the Attacker role"
