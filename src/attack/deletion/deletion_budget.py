"""Validated, finite native deletion bounds."""

from abc import ABC
from dataclasses import dataclass
from math import comb

MAX_VARIANTS = 4096


@dataclass(frozen=True)
class DeletionBudget(ABC):
    """Keep native deletion limits with a strict validation boundary.

    Attributes:
        perturbations (int): Maximum insertion count.
        positions (int): Maximum selected insertion sites.
        ascii (int): Maximum ASCII variants per site.
    """

    perturbations: int
    positions: int
    ascii: int

    def validate(self) -> None:
        """Reject unsupported or unbounded generation before inference.

        Raises:
            ValueError: If a bound is invalid or exceeds the run cap.
        """
        values: list[object] = [self.perturbations, self.positions, self.ascii]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("Deletion limits must be integers")
        if not (1 <= self.perturbations <= self.positions <= 24 and 1 <= self.ascii <= 95):
            raise ValueError("Unsupported deletion limits")
        if sum(comb(self.positions, count) * self.ascii ** count for count in range(1, self.perturbations + 1)) > MAX_VARIANTS:
            raise ValueError("Deletion limits exceed 4096 variants")
