"""Assemble the common run and assessment for a bound probe."""

from attack.envelopes.attack_envelope import AttackEnvelope
from attack.fitd_attack import FITDAttack
from attack.fitd_setup import FITDSetup
from attack.probe import Probe
from base.base_request import BaseRequest
from journal.journal import Journal
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.model_chat import ModelChat


class BoundAttackEnvelope(AttackEnvelope):
    """Share Target and final judgment wiring between bound strategies."""

    def __init__(self, base: BaseRequest, setup: FITDSetup, journal: Journal,
                 probe: Probe) -> None:
        """Compose a run over independent roles and an explicit request.

        Args:
            base (BaseRequest): Original request.
            setup (FITDSetup): Shared model resources.
            journal (Journal): Independent attempt journal.
            probe (Probe): Bounded strategy for this request.
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
                    journal, setup.text
                ),
                journal, probe=probe
            )
        )
