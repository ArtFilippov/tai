"""Read-only contract for GA's explicit finite limits."""

from typing import Protocol


class AutoDANLimits(Protocol):
    """Expose the reproducible native GA search budget."""

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
