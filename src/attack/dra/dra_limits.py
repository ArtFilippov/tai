"""Contract for finite DRA candidate limits."""

from typing import Protocol


class DRALimits(Protocol):
    """Expose the number of templates and a reproducible random seed."""

    @property
    def candidates(self) -> int:
        ...

    @property
    def seed(self) -> int:
        ...
