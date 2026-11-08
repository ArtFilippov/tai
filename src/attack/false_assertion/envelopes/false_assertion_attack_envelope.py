"""Bind FalseAssertion to its refusal-only assessment."""

from attack.attack_envelope import AttackEnvelope
from attack.attack_setup import AttackSetup
from attack.assessed_candidates import AssessedCandidates
from attack.desired_response.desired_response import DesiredResponse
from attack.desired_response.short_desired_response import ShortDesiredResponse
from attack.envelopes.bound_attack_envelope import BoundAttackEnvelope
from attack.false_assertion.false_assertion_probe import FalseAssertionProbe
from base.base_request import BaseRequest
from journal.journal import Journal
from judge.false_assertion_judge import FalseAssertionJudge
from target.model_chat import ModelChat


class FalseAssertionAttackEnvelope(AttackEnvelope):
    """Assemble independent preparation and Target chats with a refusal judge."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal,
                 source: DesiredResponse) -> None:
        """Bind one source answer and exact request to a journaled attempt.

        Args:
            base (BaseRequest): Original request with provenance.
            setup (AttackSetup): Role models.
            journal (Journal): Independent attempt journal.
            source (DesiredResponse): Provided or deferred answer.
        """
        super().__init__(
            BoundAttackEnvelope(
                base, setup, journal,
                FalseAssertionProbe(
                    setup.text,
                    ShortDesiredResponse(
                        source,
                        ModelChat(setup.attacker)
                    ),
                    ModelChat(setup.attacker),
                    journal,
                    candidates=AssessedCandidates(journal)
                ),
                judge=FalseAssertionJudge()
            )
        )
