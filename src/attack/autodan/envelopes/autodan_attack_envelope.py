"""Compose a GA run from local scoring and independent role models."""

from attack.autodan.autodan_attack import AutoDANAttack
from attack.autodan.autodan_limits import AutoDANLimits
from attack.autodan.autodan_mutation import AutoDANMutation
from attack.autodan.autodan_probe import AutoDANProbe
from attack.attack_setup import AttackSetup
from attack.attack_envelope import AttackEnvelope
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from attack.autodan.autodan_reference import AutoDANReference
from base.base_request import BaseRequest
from journal.journal import Journal
from target.logits import Logits
from target.model_chat import ModelChat


class AutoDANAttackEnvelope(AttackEnvelope):
    """Reject incompatible Target/scorer profiles before a genetic run."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal,
                 *, logits: Logits, budget: AutoDANLimits) -> None:
        """Bind native GA fitness and a separate Attacker to one request.

        Args:
            base (BaseRequest): Original request.
            setup (AttackSetup): Three role resources and bound request.
            journal (Journal): Independent run journal.
            logits (Logits): Local Qwen scoring capability.
            budget (AutoDANLimits): Validated search limits.

        """
        super().__init__(
            AutoDANAttack(
                QwenAttackEnvelope(
                    base, setup, journal,
                    AutoDANProbe(
                        logits,
                        journal,
                        budget=budget,
                        mutation=AutoDANMutation(
                            ModelChat(setup.attacker), journal
                        ),
                        corpus=AutoDANReference()
                    )
                ), setup.target, logits
            )
        )
