"""Contract for finite Bad Characters deletion limits."""

from typing import Protocol


class DeletionLimits(Protocol):
    """Expose the three native deletion generation limits."""

    @property
    def perturbations(self) -> int:
        ...

    @property
    def positions(self) -> int:
        ...

    @property
    def ascii(self) -> int:
        ...

    def validate(self) -> None:
        ...
