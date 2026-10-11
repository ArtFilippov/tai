"""Completed exchanges and opaque checkpoints belong to one conversation."""

from unittest import TestCase

from core.chats import LLMChat
from core.primitives.diagnostics.diagnostic_redactor.redaction import Redaction
from core.primitives.messages.message.text_message import TextMessage


class TestLLMChat(TestCase):
    """Check that Target and Attacker can branch independently."""

    def chat(self, system: str = "") -> LLMChat:
        return LLMChat(
            lambda turns: str(len(turns)), initial=TextMessage(system) if system else None,
            model="shared-model", system=system, redactor=Redaction(()),
        )

    def test_restore_only_changes_owner(self) -> None:
        target, attacker = self.chat(), self.chat()
        token = target.checkpoint()
        target.send(TextMessage("first"))
        attacker.send(TextMessage("other"))
        target.restore(token)
        assert (len(target.history()), len(attacker.history())) == (0, 2), (
            "Restoring Target also changed Attacker's history"
        )

    def test_reset_only_changes_owner(self) -> None:
        target, attacker = self.chat(), self.chat()
        target.send(TextMessage("first"))
        attacker.send(TextMessage("other"))
        identity = attacker.conversation_id
        target.reset()
        assert (len(target.history()), len(attacker.history()), attacker.conversation_id) == (
            0, 2, identity,
        ), "Resetting Target also changed Attacker's conversation"

    def test_foreign_checkpoint_rejected_without_mutation(self) -> None:
        target, attacker = self.chat(), self.chat()
        token = target.checkpoint()
        attacker.send(TextMessage("other"))
        with self.assertRaises(ValueError):
            attacker.restore(token)
        assert len(attacker.history()) == 2, "Foreign checkpoint changed Attacker's history"

    def test_reset_invalidates_only_own_checkpoints(self) -> None:
        target, attacker = self.chat(), self.chat()
        stale, valid = target.checkpoint(), attacker.checkpoint()
        target.reset()
        with self.assertRaises(ValueError):
            target.restore(stale)
        attacker.restore(valid)
        assert not attacker.history(), "Target reset invalidated Attacker's checkpoint"

    def test_restore_preserves_other_branches(self) -> None:
        chat = self.chat()
        first = chat.checkpoint()
        chat.send(TextMessage("first"))
        second = chat.checkpoint()
        chat.restore(first)
        chat.send(TextMessage("alternate"))
        chat.restore(second)
        assert chat.history()[0].text() == "first", "Rollback lost another saved branch"

    def test_reset_retains_own_system_turn(self) -> None:
        target = self.chat("policy")
        target.send(TextMessage("request"))
        target.reset()
        assert tuple(turn.text() for turn in target.history()) == ("policy",), (
            "Reset discarded the owning conversation's system instructions"
        )
