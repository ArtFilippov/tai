"""Bind ThreatenJSON to the original request and independent role chats."""

from attack.attack_envelope import AttackEnvelope
from attack.envelopes.bound_attack_envelope import BoundAttackEnvelope
from attack.attack_setup import AttackSetup
from attack.threaten_json.threaten_json_attack import ThreatenJSONAttack
from attack.threaten_json.threaten_json_probe import ThreatenJSONProbe
from base.base_request import BaseRequest
from journal.journal import Journal
from judge.json_answer_judge import JSONAnswerJudge
from judge.qwen_judge import QwenJudge
from target.model_chat import ModelChat


class ThreatenJSONAttackEnvelope(AttackEnvelope):
    """Compose isolated Target exchanges and a separate JSON answer judge."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal) -> None:
        """Assemble a bound run with the original unmodified request.

        Args:
            base (BaseRequest): Original request.
            setup (AttackSetup): Shared history-free model resources.
            journal (Journal): Independent attempt journal.
        """
        super().__init__(
            ThreatenJSONAttack(
                BoundAttackEnvelope(
                    base, setup, journal,
                    ThreatenJSONProbe(
                        setup.text, journal
                    ),
                    judge=JSONAnswerJudge(
                        QwenJudge(
                            setup.text,
                            ModelChat(setup.judge)
                        )
                    )
                )
            )
        )
