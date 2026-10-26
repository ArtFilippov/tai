"""Bind ThreatenJSON to the original request and independent role chats."""

from attack.envelopes.attack_envelope import AttackEnvelope
from attack.fitd_attack import FITDAttack
from attack.fitd_setup import FITDSetup
from attack.threaten_json_attack import ThreatenJSONAttack
from attack.threaten_json_probe import ThreatenJSONProbe
from base.base_request import BaseRequest
from journal.journal import Journal
from judge.json_answer_judge import JSONAnswerJudge
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.model_chat import ModelChat


class ThreatenJSONAttackEnvelope(AttackEnvelope):
    """Compose isolated Target exchanges and a separate JSON answer judge."""

    def __init__(self, base: BaseRequest, setup: FITDSetup, journal: Journal) -> None:
        """Assemble a bound run with the original unmodified request.

        Args:
            base (BaseRequest): Original request.
            setup (FITDSetup): Shared history-free model resources.
            journal (Journal): Independent attempt journal.
        """
        super().__init__(
            ThreatenJSONAttack(
                FITDAttack(
                    base,
                    ModelChat(setup.target),
                    RecordedJudge(
                        JSONAnswerJudge(
                            QwenJudge(
                                setup.text,
                                ModelChat(setup.judge)
                            )
                        ),
                        journal, setup.text
                    ),
                    journal,
                    probe=ThreatenJSONProbe(
                        setup.text, journal
                    )
                )
            )
        )
