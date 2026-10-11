"""Completed exchanges and opaque checkpoints belong to one conversation."""

from unittest import TestCase

from core.chats.envelopes.conversation import LLMChat
from core.primitives.diagnostics.diagnostic_redactor.redaction import Redaction
from core.primitives.messages.message.text_message import TextMessage


class TestLLMChat(TestCase):
    """Check that Target and Attacker can branch independently."""

    def chat(self, system: str = "") -> LLMChat:
        return LLMChat(
            lambda turns: str(len(turns)), initial=TextMessage(system) if system else None,
            model="shared-model", system=system, redactor=Redaction(()),
        )

    def shared(self) -> tuple[LLMChat, LLMChat]:
        target, attacker = self.chat(), self.chat()
        target.send(TextMessage("target request"))
        attacker.send(TextMessage("attacker request"))
        return target, attacker

    def refuses(self, chat: LLMChat, checkpoint: object) -> None:
        with self.assertRaises(ValueError):
            chat.restore(checkpoint)

    def alternate(self, chat: LLMChat, checkpoint: object) -> None:
        chat.restore(checkpoint)
        chat.send(TextMessage("alternate"))

    def branch(self, chat: LLMChat) -> tuple[object, object]:
        first = chat.checkpoint()
        chat.send(TextMessage("first"))
        second = chat.checkpoint()
        return first, second

    def test_restore_only_changes_owner(self) -> None:
        target, attacker = self.shared()
        token = self.branch(target)[0]
        target.send(TextMessage("later"))
        target.restore(token)
        assert (len(target.history()), len(attacker.history())) == (2, 2), (
            "Restoring Target also changed Attacker's history"
        )

    def test_reset_only_changes_owner(self) -> None:
        target, attacker = self.shared()
        identity, history = attacker.conversation_id, attacker.history()
        target.reset()
        assert (target.history(), attacker.history(), attacker.conversation_id) == (
            (), history, identity,
        ), "Resetting Target also changed Attacker's conversation"

    def test_foreign_checkpoint_rejected_without_mutation(self) -> None:
        target = self.chat()
        attacker = self.chat("policy")
        attacker.send(TextMessage("other"))
        self.refuses(attacker, target.checkpoint())
        assert len(attacker.history()) == 3, "Foreign checkpoint changed Attacker's history"

    def test_reset_invalidates_only_own_checkpoints(self) -> None:
        target, attacker = self.chat("target policy"), self.chat()
        stale, valid = target.checkpoint(), attacker.checkpoint()
        target.reset()
        self.refuses(target, stale)
        attacker.restore(valid)
        assert not attacker.history(), "Target reset invalidated Attacker's checkpoint"

    def test_restore_preserves_other_branches(self) -> None:
        chat = self.chat()
        first, second = self.branch(chat)
        self.alternate(chat, first)
        chat.restore(second)
        assert chat.history()[0].text() == "first", "Rollback lost another saved branch"

    def test_reset_retains_own_system_turn(self) -> None:
        target = self.chat("policy")
        identity = target.conversation_id
        target.send(TextMessage("request"))
        target.reset()
        outcome = (
            tuple(turn.text() for turn in target.history()), target.conversation_id != identity,
        )
        assert outcome == (
            ("policy",), True,
        ), "Reset discarded the system instructions or failed to rotate identity"
