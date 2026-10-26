"""Bind a FITD attack to shared model resources and one request."""

from attack.envelopes.attack_envelope import AttackEnvelope
from attack.fitd_attack import FITDAttack
from attack.fitd_probe import FITDProbe
from attack.fitd_setup import FITDSetup
from base.base_request import BaseRequest
from journal.journal import Journal
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.continuing_chat import ContinuingChat
from target.model_chat import ModelChat


class FITDAttackEnvelope(AttackEnvelope):
    """Bind one request and isolated role conversations to shared models."""

    def __init__(self, base: BaseRequest, setup: FITDSetup, journal: Journal,
                 turns: int) -> None:
        """Assemble isolated FITD conversations for this attempt.

        Args:
            base (BaseRequest): Original request.
            setup (FITDSetup): Shared model resources and request text.
            journal (Journal): Attempt event sink.
            turns (int): Maximum Target exchanges.
        """
        super().__init__(
            FITDAttack(
                base,
                ModelChat(setup.target),
                RecordedJudge(
                    QwenJudge(
                        setup.text,
                        ModelChat(setup.judge)
                    ),
                    journal,
                    setup.text
                ),
                journal,
                probe=FITDProbe(
                    setup.text,
                    ContinuingChat(
                        ModelChat(setup.attacker)
                    ),
                    journal, turns
                )
            )
        )
