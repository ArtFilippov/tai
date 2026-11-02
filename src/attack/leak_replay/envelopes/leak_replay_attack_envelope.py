"""Bind a masking conversation and desired answer to one attack."""

from attack.attack_envelope import AttackEnvelope
from attack.attack_setup import AttackSetup
from attack.desired_response.desired_response import DesiredResponse
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from attack.leak_replay.leak_replay_probe import LeakReplayProbe
from base.base_request import BaseRequest
from journal.journal import Journal
from target.model_chat import ModelChat


class LeakReplayAttackEnvelope(AttackEnvelope):
    """Assemble independent masking, Target and judgment conversations."""

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
                LeakReplayProbe(
                    setup.text, source,
                    ModelChat(setup.attacker),
                    journal
                )
            )
        )
