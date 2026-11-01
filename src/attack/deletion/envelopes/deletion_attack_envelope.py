"""Bind deletion to a supplied base and independent journal."""

from attack.assessed_candidates import AssessedCandidates
from attack.attack_setup import AttackSetup
from attack.deletion.deletion_attack import DeletionAttack
from attack.deletion.deletion_limits import DeletionLimits
from attack.deletion.deletion_native import NativeDeletion
from attack.deletion.deletion_probe import DeletionProbe
from attack.attack_envelope import AttackEnvelope
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from base.base_request import BaseRequest
from journal.journal import Journal


class DeletionAttackEnvelope(AttackEnvelope):
    """Compose separate Target and judgment conversations for deletion."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal, budget: DeletionLimits) -> None:
        """Bind shared role resources to a single original request.

        Args:
            base (BaseRequest): Supplied request.
            setup (AttackSetup): History-free role resources.
            journal (Journal): Independent run sink.
            budget (DeletionLimits): Native generation limits.
        """
        super().__init__(
            DeletionAttack(
                QwenAttackEnvelope(
                    base, setup, journal,
                    DeletionProbe(
                        setup.text,
                        AssessedCandidates(journal),
                        journal, budget=budget,
                        variants=NativeDeletion(
                            budget.perturbations, budget.positions, budget.ascii
                        )
                    )
                )
            )
        )
