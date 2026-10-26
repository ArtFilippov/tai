"""Forward attack operations to a composed delegate."""

from typing import override

from attack.attack import Attack


class AttackEnvelope(Attack):
    """Preserve the Attack interface for composed attack variants."""

    def __init__(self, delegate: Attack) -> None:
        """Hold the attack receiving every operation.

        Args:
            delegate (Attack): Bound attack.
        """
        self.delegate = delegate

    @override
    def description(self) -> dict[str, object]:
        return self.delegate.description()

    @override
    def run(self) -> str | None:
        return self.delegate.run()
