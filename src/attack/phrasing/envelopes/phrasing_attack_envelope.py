"""Compose one tense's independent candidate conversations and fulfillment Judge."""

from attack.assessed_candidates import AssessedCandidates
from attack.attack_envelope import AttackEnvelope
from attack.attack_setup import AttackSetup
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from attack.phrasing.phrasing_plan import PhrasingPlan
from attack.phrasing.phrasing_probe import PhrasingProbe
from base.base_request import BaseRequest
from journal.journal import Journal
from target.model_chat import ModelChat
from target.sampled_model import SampledModel
from target.seeded_sampling import SeededSampling


class PhrasingAttackEnvelope(AttackEnvelope):
    """Share weights while isolating this tense's decoding, conversations and result."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal, *, plan: PhrasingPlan) -> None:
        """Bind one tense to a new seed-zero Attacker stream and ordinary Judge.

        Args:
            base (BaseRequest): Original request and provenance.
            setup (AttackSetup): Shared role resources bound to that request.
            journal (Journal): Independent event sink.
            plan (PhrasingPlan): Tense, pinned instruction and preparation budget.
        """
        super().__init__(
            QwenAttackEnvelope(
                base, setup, journal,
                PhrasingProbe(
                    base,
                    AssessedCandidates(journal),
                    journal,
                    ModelChat(
                        SampledModel(
                            setup.attacker,
                            SeededSampling()
                        )
                    ),
                    plan=plan
                )
            )
        )
