"""Contract for a FITD run assembled from independently supplied roles."""

from typing import Protocol

from fitd.attack import Attack


class FITDAttackRun(Attack, Protocol):
    """Selects a result and requests its assessment for one FITD run."""
