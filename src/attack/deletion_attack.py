"""Public Bad Characters deletion attack."""

from dataclasses import dataclass
from typing import override

from attack.attack import Attack


@dataclass(eq=False)
class DeletionAttack(Attack):
    """Expose the bound deletion run through the Attack interface.

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
