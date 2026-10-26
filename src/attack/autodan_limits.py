"""Read-only contract for AutoDAN's explicit finite limits."""

from typing import Protocol


class AutoDANLimits(Protocol):
    """Expose the reproducible native GA/HGA search budget."""

    @property
    def steps(self) -> int: ...

    @property
    def batch(self) -> int: ...

    @property
    def elites(self) -> int: ...

    @property
    def points(self) -> int: ...

    @property
    def crossover(self) -> float: ...

    @property
    def mutation(self) -> float: ...

    @property
    def seed(self) -> int: ...

    @property
    def target(self) -> str: ...

    @property
    def mode(self) -> str: ...
