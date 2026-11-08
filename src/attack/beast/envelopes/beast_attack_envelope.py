"""Bind local BEAST search and the ordinary independent answer judge."""

from attack.attack_envelope import AttackEnvelope
from attack.attack_setup import AttackSetup
from attack.probe import Probe
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from base.base_request import BaseRequest
from journal.journal import Journal


class BeastAttackEnvelope(AttackEnvelope):
    """Compose one search, final Target conversation and recorded judgment."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal,
                 probe: Probe) -> None:
        """Bind one exact request to local scoring and its independent result.

        Args:
            base (BaseRequest): Original request and provenance.
            setup (AttackSetup): Target and judge roles.
            journal (Journal): Independent attempt journal.
            probe (Probe): Request-bound local search and answer.
        """
        super().__init__(
            QwenAttackEnvelope(
                base, setup, journal,
                probe
            )
        )
