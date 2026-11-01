"""Bind a DRA request to role resources and one independent journal."""

from attack.assessed_candidates import AssessedCandidates
from attack.attack_setup import AttackSetup
from attack.dra.dra_attack import DRAAttack
from attack.dra.dra_budget import DRABudget
from attack.dra.dra_probe import DRAProbe
from attack.attack_envelope import AttackEnvelope
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from base.base_request import BaseRequest
from journal.journal import Journal


class DRAAttackEnvelope(AttackEnvelope):
    """Compose independent Target and judge conversations for a DRA run."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal, budget: DRABudget) -> None:
        """Bind an exact request and validated budget to fresh conversations.

        Args:
            base (BaseRequest): Supplied request.
            setup (AttackSetup): Shared history-free role resources.
            journal (Journal): Independent run event sink.
            budget (DRABudget): Candidate count and seed.
        """
        super().__init__(
            DRAAttack(
                QwenAttackEnvelope(
                    base, setup, journal,
                    DRAProbe(
                        setup.text,
                        AssessedCandidates(journal),
                        journal, budget
                    )
                )
            )
        )
