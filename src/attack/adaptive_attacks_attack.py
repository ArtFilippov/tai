"""Public adaptive attack over a journaled, bound run."""

from dataclasses import dataclass
from typing import override

from attack.attack import Attack


@dataclass(eq=False)
class AdaptiveAttacksAttack(Attack):
    """Expose one adaptive attempt through the shared Attack interface.

    Attributes:
        delegate (Attack): Bound and journaled attempt.
    """

    delegate: Attack

    @override
    def description(self) -> dict[str, object]:
        return self.delegate.description()

    @override
    def run(self) -> str | None:
        return self.delegate.run()
