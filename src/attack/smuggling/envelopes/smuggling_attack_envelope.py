"""Bind deterministic Smuggling to the project's fulfillment Judge."""

from collections.abc import Sequence

from attack.assessed_candidates import AssessedCandidates
from attack.attack_envelope import AttackEnvelope
from attack.attack_setup import AttackSetup
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from attack.smuggling.smuggling_probe import SmugglingProbe
from base.base_request import BaseRequest
from journal.journal import Journal


class SmugglingAttackEnvelope(AttackEnvelope):
    """Compose one transformed request with independent Target and Judge chats."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal, *,
                 techniques: Sequence[str] = ("homoglyph",)) -> None:
        """Bind the original request, role resources and attempt journal.

        Args:
            base (BaseRequest): Original request with provenance.
            setup (AttackSetup): Target and fulfillment Judge model resources.
            journal (Journal): Independent run journal.
            techniques (Sequence[str], default=("homoglyph",)): Nonempty homoglyph-only selection.
        """
        super().__init__(
            QwenAttackEnvelope(
                base, setup, journal,
                SmugglingProbe(
                    base, AssessedCandidates(journal), journal,
                    techniques=techniques
                )
            )
        )
