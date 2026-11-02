"""Bind a prepared answer to a journaled Slur Completion attempt."""

from attack.attack_envelope import AttackEnvelope
from attack.attack_setup import AttackSetup
from attack.desired_response.desired_response import DesiredResponse
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from attack.slur_completion.slur_completion_probe import SlurCompletionProbe
from base.base_request import BaseRequest
from journal.journal import Journal


class SlurCompletionAttackEnvelope(AttackEnvelope):
    """Assemble one prefilled Target exchange and independent judgment."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal,
                 source: DesiredResponse) -> None:
        """Bind an original request and injected answer to an attempt.

        Args:
            base (BaseRequest): Original request with provenance.
            setup (AttackSetup): Independent role models.
            journal (Journal): Attempt journal.
            source (DesiredResponse): Prepared answer source.
        """
        super().__init__(
            QwenAttackEnvelope(
                base, setup, journal,
                SlurCompletionProbe(
                    setup.text, source, journal
                )
            )
        )
