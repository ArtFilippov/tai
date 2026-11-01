"""Assemble the common run and assessment for a bound probe."""

from attack.attack_run import AttackRun
from attack.attack_setup import AttackSetup
from attack.attack_envelope import AttackEnvelope
from attack.probe import Probe
from base.base_request import BaseRequest
from journal.journal import Journal
from judge.judge import Judge
from judge.recorded_judge import RecordedJudge
from target.model_chat import ModelChat


class BoundAttackEnvelope(AttackEnvelope):
    """Share Target and recorded judgment wiring between bound strategies."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal,
                 probe: Probe, *, judge: Judge) -> None:
        """Compose a run over independent roles and an explicit request.

        Args:
            base (BaseRequest): Original request.
            setup (AttackSetup): Shared model resources.
            journal (Journal): Independent attempt journal.
            probe (Probe): Bounded strategy for this request.
            judge (Judge): Strategy-specific assessment of presented text.
        """
        super().__init__(
            AttackRun(
                base,
                ModelChat(setup.target),
                RecordedJudge(
                    judge,
                    journal, setup.text
                ),
                journal, probe=probe
            )
        )
