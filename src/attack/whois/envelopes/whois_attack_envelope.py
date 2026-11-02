"""Bind one WHOIS injection to its original base and journal."""

from attack.attack_envelope import AttackEnvelope
from attack.attack_setup import AttackSetup
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from attack.whois.whois_attack import WhoisAttack
from attack.whois.whois_probe import WhoisProbe
from base.base_request import BaseRequest
from journal.journal import Journal


class WhoisAttackEnvelope(AttackEnvelope):
    """Compose isolated Target and judge conversations for a WHOIS attempt."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal) -> None:
        """Bind the original request and history-free role resources.

        Args:
            base (BaseRequest): Original request and provenance.
            setup (AttackSetup): Target and judge resources.
            journal (Journal): Independent attempt journal.
        """
        super().__init__(
            WhoisAttack(
                QwenAttackEnvelope(
                    base, setup, journal,
                    WhoisProbe(
                        setup.text, journal
                    )
                )
            )
        )
