"""Validated bounded genetic search configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

from attacks.shared.contracts import AUTODAN, StrategyConfiguration

PARAMETERS = frozenset({"expected", "num_steps", "batch_size", "num_elites", "crossover_rate", "num_points", "mutation_rate", "seed"})


class AutoDANSearch(Protocol):
    def __call__(self) -> str: ...
    def score(self, prefixes: list[str], step: int) -> list[float]: ...
    def mutate(self, prefix: str, step: int) -> str: ...


class AutoDANOptions(TypedDict, total=False):
    expected: str
    num_steps: int
    batch_size: int
    num_elites: int
    crossover_rate: float
    num_points: int
    mutation_rate: float
    seed: int


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"autodan.{name}: требуется целое число >= {minimum}")
    return value


def _rate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"autodan.{name}: требуется число от 0 до 1")
    return float(value)


@dataclass(frozen=True)
class Evolution:
    num_elites: int = 2
    crossover_rate: float = 0.5
    num_points: int = 5
    mutation_rate: float = 0.1
    seed: int = 0


@dataclass(frozen=True)
class AutoDANProfile:
    expected: str
    num_steps: int = 25
    batch_size: int = 8
    evolution: Evolution = Evolution()

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.expected), str) or not self.expected.strip():
            raise ValueError("autodan.expected: требуется непустое ожидаемое продолжение")
        _ = _integer(self.num_steps, "num_steps")
        _ = _integer(self.batch_size, "batch_size", minimum=2)
        values = self.evolution
        _ = _integer(values.num_elites, "num_elites")
        if values.num_elites >= self.batch_size:
            raise ValueError("autodan.num_elites: должно быть меньше batch_size")
        _ = _integer(values.num_points, "num_points")
        _ = _integer(values.seed, "seed", minimum=0)
        _ = _rate(values.crossover_rate, "crossover_rate")
        _ = _rate(values.mutation_rate, "mutation_rate")

    def configuration(self) -> StrategyConfiguration:
        return StrategyConfiguration(AUTODAN, {"expected": self.expected, "num_steps": self.num_steps,
                                               "batch_size": self.batch_size, **vars(self.evolution), "generations": 1}, self.num_steps)


def autodan_profile(parameters: Mapping[str, object]) -> AutoDANProfile:
    return AutoDANProfile(
        expected=cast(str, parameters.get("expected")), num_steps=cast(int, parameters.get("num_steps", 25)),
        batch_size=cast(int, parameters.get("batch_size", 8)), evolution=Evolution(
            num_elites=cast(int, parameters.get("num_elites", 2)),
            crossover_rate=cast(float, parameters.get("crossover_rate", 0.5)),
            num_points=cast(int, parameters.get("num_points", 5)), mutation_rate=cast(float, parameters.get("mutation_rate", 0.1)),
            seed=cast(int, parameters.get("seed", 0)),
        ),
    )
