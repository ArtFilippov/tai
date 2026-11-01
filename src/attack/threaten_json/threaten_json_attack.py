"""Public ThreatenJSON attack over a bound journaled run."""

from dataclasses import dataclass
from typing import override

from attack.attack import Attack


@dataclass(eq=False)
class ThreatenJSONAttack(Attack):
    """Expose a single ThreatenJSON attempt through the public Attack interface.

    Attributes:
        delegate (Attack): Fully bound run.
    """

    delegate: Attack

    @override
    def description(self) -> dict[str, object]:
        return self.delegate.description()

    @override
    def run(self) -> str | None:
        return self.delegate.run()
