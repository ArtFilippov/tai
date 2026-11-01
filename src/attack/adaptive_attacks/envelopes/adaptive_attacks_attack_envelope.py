"""Bind one adaptive goal to shared model resources and an independent journal."""

from attack.adaptive_attacks.adaptive_attacks_attack import AdaptiveAttacksAttack
from attack.attack_setup import AttackSetup
from attack.adaptive_attacks.adaptive_attacks_probe import AdaptiveAttacksProbe
from attack.assessed_candidates import AssessedCandidates
from attack.attack_envelope import AttackEnvelope
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from base.base_request import BaseRequest
from journal.journal import Journal


class AdaptiveAttacksAttackEnvelope(AttackEnvelope):
    """Compose fresh Target and judge chats over shared history-free weights."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal) -> None:
        """Bind the original request and one journal to adaptive candidates.

        Args:
            base (BaseRequest): Supplied original request.
            setup (AttackSetup): Shared history-free model resources.
            journal (Journal): Independent run event sink.
        """
        super().__init__(
            AdaptiveAttacksAttack(
                QwenAttackEnvelope(
                    base, setup, journal,
                    AdaptiveAttacksProbe(
                        setup.text,
                        AssessedCandidates(journal)
                    )
                )
            )
        )
