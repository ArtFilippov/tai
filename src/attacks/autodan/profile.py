"""Validated bounded genetic search configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, cast

from attacks.shared.contracts import AUTODAN, StrategyConfiguration

PARAMETERS = frozenset({"expected", "mode", "num_steps", "batch_size", "num_elites", "crossover_rate", "num_points", "mutation_rate", "seed"})
type SearchMode = Literal["ga", "hga"]
HGA: SearchMode = "hga"
GA: SearchMode = "ga"
POINTS = "num_points"


class AutoDANSearch(Protocol):
    def __call__(self) -> str: ...
    def score(self, prefixes: list[str], step: int) -> list[float]: ...
    def mutate(self, prefix: str, step: int) -> str: ...


class RandomSource(Protocol):
    def random(self) -> float: ...


class HierarchicalActions(RandomSource, Protocol):
    def mutate(self, prefix: str) -> str: ...
    def check_running(self) -> None: ...


class AutoDANOptions(TypedDict, total=False):
    expected: str
    mode: SearchMode
    num_steps: int
    batch_size: int
    num_elites: int
    crossover_rate: float
    num_points: int
    mutation_rate: float
    seed: int


def _integer(value: object, name: str, *, minimum: int = 1, maximum: int = 100) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"autodan.{name}: требуется целое число от {minimum} до {maximum}")
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
    mode: SearchMode = GA
    num_steps: int = 25
    batch_size: int = 8
    evolution: Evolution = Evolution()

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.expected), str) or not self.expected.strip():
            raise ValueError("autodan.expected: требуется непустое ожидаемое продолжение")
        if self.mode not in (GA, HGA):
            raise ValueError("autodan.mode: поддерживается ga или hga")
        _ = _integer(self.num_steps, "num_steps")
        _ = _integer(self.batch_size, "batch_size", minimum=2, maximum=64)
        values = self.evolution
        _ = _integer(values.num_elites, "num_elites", maximum=63)
        if values.num_elites >= self.batch_size:
            raise ValueError("autodan.num_elites: должно быть меньше batch_size")
        _ = _integer(values.num_points, "num_points", maximum=32)
        _ = _integer(values.seed, "seed", minimum=0, maximum=2**32 - 1)
        _ = _rate(values.crossover_rate, "crossover_rate")
        _ = _rate(values.mutation_rate, "mutation_rate")

    def configuration(self) -> StrategyConfiguration:
        evolution = vars(self.evolution).copy()
        if self.mode == HGA:
            del evolution[POINTS]
        return StrategyConfiguration(AUTODAN, {"expected": self.expected, "mode": self.mode, "num_steps": self.num_steps,
                                               "batch_size": self.batch_size, **evolution, "generations": 1}, self.num_steps)


def autodan_profile(parameters: Mapping[str, object]) -> AutoDANProfile:
    if parameters.get("mode", GA) == HGA and POINTS in parameters:
        raise ValueError("autodan.num_points: используется только в режиме ga")
    return AutoDANProfile(
        expected=cast(str, parameters.get("expected")), mode=cast(SearchMode, parameters.get("mode", GA)),
        num_steps=cast(int, parameters.get("num_steps", 25)),
        batch_size=cast(int, parameters.get("batch_size", 8)), evolution=Evolution(
            num_elites=cast(int, parameters.get("num_elites", 2)),
            crossover_rate=cast(float, parameters.get("crossover_rate", 0.5)),
            num_points=cast(int, parameters.get("num_points", 5)), mutation_rate=cast(float, parameters.get("mutation_rate", 0.1)),
            seed=cast(int, parameters.get("seed", 0)),
        ),
    )
