"""Bind one GOAT request to separate role conversations."""

from attack.envelopes.attack_envelope import AttackEnvelope
from attack.envelopes.bound_attack_envelope import BoundAttackEnvelope
from attack.fitd_setup import FITDSetup
from attack.goat_attack import GOATAttack
from attack.goat_probe import GOATProbe
from base.base_request import BaseRequest
from journal.journal import Journal
from target.model_chat import ModelChat


class GOATAttackEnvelope(AttackEnvelope):
    """Compose one independent Attacker, Target, and final judge per run."""

    def __init__(self, base: BaseRequest, setup: FITDSetup, journal: Journal,
                 turns: int) -> None:
        """Bind a native GOAT probe to independent role resources.

        Args:
            base (BaseRequest): Original request.
            setup (FITDSetup): Shared history-free role resources.
            journal (Journal): Attempt journal.
            turns (int): Maximum Target exchanges.
        """
        super().__init__(
            GOATAttack(
                BoundAttackEnvelope(
                    base, setup, journal,
                    GOATProbe(
                        setup.text,
                        ModelChat(setup.attacker),
                        journal, turns
                    )
                )
            )
        )
