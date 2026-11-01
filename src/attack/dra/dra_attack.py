"""Public DRA attack over a bound journaled run."""

from dataclasses import dataclass
from typing import override

from attack.attack import Attack


@dataclass(eq=False)
class DRAAttack(Attack):
    """Expose a composed DRA attempt through the Attack interface.

    Attributes:
        delegate (Attack): Bound attempt.
    """

    delegate: Attack

    @override
    def description(self) -> dict[str, object]:
        return self.delegate.description()

    @override
    def run(self) -> str | None:
        return self.delegate.run()
