"""Bind selected pinned DAN sections to fulfillment assessment."""

from collections.abc import Sequence

from attack.attack_envelope import AttackEnvelope
from attack.attack_setup import AttackSetup
from attack.dan.dan_probe import DanProbe
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from base.base_request import BaseRequest
from journal.journal import Journal
from prompt_collection.ablation_dan_prompts import AblationDanPrompts
from prompt_collection.dan_prompts import DanPrompts
from prompt_collection.named_dan_prompts import NamedDanPrompts
from prompt_collection.provided_prompt import ProvidedPrompt
from prompt_collection.target_dan_text import TargetDanText
from prompt_collection.wild_dan_prompts import WildDanPrompts


class DanAttackEnvelope(AttackEnvelope):
    """Compose one catalog traversal with independent Target and judge chats."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal, *,
                 sections: Sequence[str] = ("named", "ablation", "wild")) -> None:
        """Bind an exact request to the selected section and recorded Qwen judge.

        Args:
            base (BaseRequest): Original request and provenance.
            setup (AttackSetup): Configured Target and fulfillment judge resources.
            journal (Journal): Independent journal for the complete traversal.
            sections (Sequence[str], default=("named", "ablation", "wild")): Nonempty section subset.
        """
        super().__init__(
            QwenAttackEnvelope(
                base, setup, journal,
                DanProbe(
                    base,
                    DanPrompts(
                        {
                            "named": NamedDanPrompts(
                                TargetDanText(setup.target),
                                lambda text, identity: ProvidedPrompt(text, identity)
                            ),
                            "ablation": AblationDanPrompts(
                                TargetDanText(setup.target),
                                lambda text, identity: ProvidedPrompt(text, identity)
                            ),
                            "wild": WildDanPrompts(
                                TargetDanText(setup.target),
                                lambda text, identity: ProvidedPrompt(text, identity)
                            )
                        },
                        sections=sections
                    ),
                    journal
                )
            )
        )
