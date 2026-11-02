"""Bind one Grandma roleplay to its original request and journal."""

from attack.attack_envelope import AttackEnvelope
from attack.attack_setup import AttackSetup
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from attack.grandma.grandma_attack import GrandmaAttack
from attack.grandma.grandma_probe import GrandmaProbe
from base.base_request import BaseRequest
from journal.journal import Journal


class GrandmaAttackEnvelope(AttackEnvelope):
    """Compose isolated Target and judge conversations for a Grandma attempt."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal) -> None:
        """Bind the exact request and independent attempt journal.

        Args:
            base (BaseRequest): Original request and provenance.
            setup (AttackSetup): Target and judge resources.
            journal (Journal): Independent attempt journal.
        """
        super().__init__(
            GrandmaAttack(
                QwenAttackEnvelope(
                    base, setup, journal,
                    GrandmaProbe(
                        setup.text, journal
                    )
                )
            )
        )
