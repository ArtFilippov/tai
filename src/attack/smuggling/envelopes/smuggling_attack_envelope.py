"""Bind deterministic Smuggling to the project's fulfillment Judge."""

from collections.abc import Sequence

from attack.assessed_candidates import AssessedCandidates
from attack.attack_envelope import AttackEnvelope
from attack.attack_setup import AttackSetup
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from attack.smuggling.smuggling_probe import SmugglingProbe
from base.base_request import BaseRequest
from journal.journal import Journal
from target.model_chat import ModelChat


class SmugglingAttackEnvelope(AttackEnvelope):
    """Compose selected techniques with independent Attacker, Target and Judge chats."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal, *,
                 techniques: Sequence[str] = ("hypothetical", "homoglyph")) -> None:
        """Bind the original request, role resources and attempt journal.

        Args:
            base (BaseRequest): Original request with provenance.
            setup (AttackSetup): Attacker, Target and fulfillment Judge resources.
            journal (Journal): Independent run journal.
            techniques (Sequence[str], default=("hypothetical", "homoglyph")): Nonempty supported selection.
        """
        super().__init__(
            QwenAttackEnvelope(
                base, setup, journal,
                SmugglingProbe(
                    base, AssessedCandidates(journal), journal,
                    ModelChat(setup.attacker),
                    techniques=techniques
                )
            )
        )
