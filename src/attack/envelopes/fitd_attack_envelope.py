"""Bind FITD to shared model resources and one request."""

from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from attack.attack_setup import AttackSetup
from attack.fitd_probe import FITDProbe
from base.base_request import BaseRequest
from journal.journal import Journal
from target.continuing_chat import ContinuingChat
from target.model_chat import ModelChat


class FITDAttackEnvelope(QwenAttackEnvelope):
    """Bind one request and isolated role conversations to shared models."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal,
                 turns: int) -> None:
        """Assemble isolated FITD conversations for this attempt.

        Args:
            base (BaseRequest): Original request.
            setup (AttackSetup): Shared model resources and request text.
            journal (Journal): Attempt event sink.
            turns (int): Maximum Target exchanges.
        """
        super().__init__(
            base, setup, journal,
            FITDProbe(
                setup.text,
                ContinuingChat(
                    ModelChat(setup.attacker)
                ),
                journal, turns
            )
        )
