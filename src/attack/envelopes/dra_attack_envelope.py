"""Bind a DRA request to role resources and one independent journal."""

from attack.assessed_candidates import AssessedCandidates
from attack.dra_attack import DRAAttack
from attack.dra_budget import DRABudget
from attack.dra_probe import DRAProbe
from attack.envelopes.attack_envelope import AttackEnvelope
from attack.envelopes.bound_attack_envelope import BoundAttackEnvelope
from attack.fitd_setup import FITDSetup
from base.base_request import BaseRequest
from journal.journal import Journal


class DRAAttackEnvelope(AttackEnvelope):
    """Compose independent Target and judge conversations for a DRA run."""

    def __init__(self, base: BaseRequest, setup: FITDSetup, journal: Journal, budget: DRABudget) -> None:
        """Bind an exact request and validated budget to fresh conversations.

        Args:
            base (BaseRequest): Supplied request.
            setup (FITDSetup): Shared history-free role resources.
            journal (Journal): Independent run event sink.
            budget (DRABudget): Candidate count and seed.
        """
        super().__init__(
            DRAAttack(
                BoundAttackEnvelope(
                    base, setup, journal,
                    DRAProbe(
                        setup.text,
                        AssessedCandidates(journal),
                        journal, budget
                    )
                )
            )
        )
