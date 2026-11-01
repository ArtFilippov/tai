"""Bind the common Qwen answer judgment to an attack probe."""

from attack.attack_setup import AttackSetup
from attack.envelopes.bound_attack_envelope import BoundAttackEnvelope
from attack.probe import Probe
from base.base_request import BaseRequest
from journal.journal import Journal
from judge.qwen_judge import QwenJudge
from target.model_chat import ModelChat


class QwenAttackEnvelope(BoundAttackEnvelope):
    """Assemble a recorded attack with the usual answer judge."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal,
                 probe: Probe) -> None:
        """Bind one original request to independent Target and judge chats.

        Args:
            base (BaseRequest): Original request.
            setup (AttackSetup): Role models and exact request.
            journal (Journal): Independent attempt journal.
            probe (Probe): Strategy for this attempt.
        """
        super().__init__(
            base, setup, journal, probe,
            judge=QwenJudge(
                setup.text,
                ModelChat(setup.judge)
            )
        )
