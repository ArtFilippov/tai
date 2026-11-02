"""Public WHOIS attack over a bound journaled run."""

from dataclasses import dataclass
from typing import override

from attack.attack import Attack


@dataclass(eq=False)
class WhoisAttack(Attack):
    """Expose one WHOIS attempt through the existing Attack interface.

    Attributes:
        delegate (Attack): Fully bound attempt.
    """

    delegate: Attack

    @override
    def description(self) -> dict[str, object]:
        return self.delegate.description()

    @override
    def run(self) -> str | None:
        return self.delegate.run()
