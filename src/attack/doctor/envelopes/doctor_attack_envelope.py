"""Bind selected Doctor techniques to the project's fulfillment Judge."""

from collections.abc import Sequence

from attack.assessed_candidates import AssessedCandidates
from attack.attack_envelope import AttackEnvelope
from attack.attack_setup import AttackSetup
from attack.doctor.doctor_probe import DoctorProbe
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from base.base_request import BaseRequest
from journal.journal import Journal


class DoctorAttackEnvelope(AttackEnvelope):
    """Compose independent Target exchanges and full-response fulfillment assessment."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal, *,
                 techniques: Sequence[str] = ("puppetry", "bypass", "bypass_leet"), seed: int = 0) -> None:
        """Bind the original request, role resources and independent journal.

        Args:
            base (BaseRequest): Original request with provenance.
            setup (AttackSetup): Target and Judge resources bound to the request.
            journal (Journal): Independent attempt journal.
            techniques (Sequence[str], default=("puppetry", "bypass", "bypass_leet")): Nonempty technique selection.
            seed (int, default=0): Run-local leetspeak seed.
        """
        super().__init__(
            QwenAttackEnvelope(
                base, setup, journal,
                DoctorProbe(
                    base, AssessedCandidates(journal), journal,
                    techniques=techniques, seed=seed
                )
            )
        )
