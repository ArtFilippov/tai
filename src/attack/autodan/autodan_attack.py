"""Public AutoDAN GA attack bound to one journaled attempt."""

from dataclasses import dataclass
from typing import override

from attack.attack import Attack
from target.logits import Logits
from target.model import Model

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(eq=False)
class AutoDANAttack(Attack):
    """Guard a bound GA run against incompatible local Target backends.

    Attributes:
        delegate (Attack): Bound journaled run.
        target (Model): Model supplying real Target conversations.
        logits (Logits): Dedicated independent fitness capability.
    """

    delegate: Attack
    target: Model
    logits: Logits

    @override
    def description(self) -> dict[str, object]:
        return self.delegate.description()

    @override
    def run(self) -> str | None:
        profile = self.logits.profile()
        settings = self.target.settings()
        if (profile.get("id") != QWEN
                or settings.get("id") != QWEN
                or settings.get("revision") != profile.get("revision")):
            raise ValueError("AutoDAN requires the pinned local Qwen Target and scorer")
        return self.delegate.run()
