"""Shared weights receive only the supplied conversation."""

import unittest

from garak.attempt import Conversation, Message, Turn

from target.model_chat import ModelChat
from target.scripted_model import ScriptedModel


class TestModelChat(unittest.TestCase):
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
        target = ModelChat(model)
        attacker = ModelChat(model)
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

    def test_prefill_returns_complete_message_without_changing_reply(self) -> None:
        model = ScriptedModel(
            [
                [
                    {
                        "role": "user",
                        "content": "Target request"
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Other request"
                    }
                ]
            ]
        )
        target = ModelChat(model)
        result = target.prefill(
            Conversation(
                [
                    Turn(
                        "user",
                        Message("Target request")
                    )
                ]
            ),
            "Beginning"
        )
        ordinary = ModelChat(model).reply(
            Conversation(
                [
                    Turn(
                        "user",
                        Message("Other request")
                    )
                ]
            )
        )
        assert (result, ordinary) == ("BeginningReply", "Reply"), "Prefill lost its prefix or changed ordinary generation"
