"""Public GOAT attack boundary over a bound, journaled run."""

from dataclasses import dataclass
from typing import override

from attack.attack import Attack


@dataclass(eq=False)
class GOATAttack(Attack):
    """Expose one independent GOAT run through the public Attack interface.

    Attributes:
        delegate (Attack): Fully bound and journaled GOAT run.
    """

    delegate: Attack

    @override
    def description(self) -> dict[str, object]:
        return self.delegate.description()

    @override
    def run(self) -> str | None:
        return self.delegate.run()
