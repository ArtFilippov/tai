"""Validated local DRA generation budget and randomness."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TypedDict, cast

from attacks.shared.contracts import DRA, StrategyConfiguration

PARAMETERS = frozenset({"max_variants", "seed", "toxic_trunc", "benign_trunc", "benign_full_token_prob"})
TEMPLATE_COUNT = 4
FULL_TOKEN_PROBABILITY = "benign_full_token_prob"


class DRAOptions(TypedDict, total=False):
    max_variants: int
    seed: int
    toxic_trunc: float
    benign_trunc: float
    benign_full_token_prob: float


@dataclass(frozen=True)
class DRAProfile:
    max_variants: int = TEMPLATE_COUNT
    seed: int = 0
    toxic_trunc: float = 0.5
    benign_trunc: float = 0.5
    benign_full_token_prob: float = 0.33

    def __post_init__(self) -> None:
        if isinstance(self.max_variants, bool) or not isinstance(cast(object, self.max_variants), int) or not 1 <= self.max_variants <= TEMPLATE_COUNT:
            raise ValueError("max_variants must be between 1 and 4")
        if isinstance(self.seed, bool) or not isinstance(cast(object, self.seed), int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("toxic_trunc", "benign_trunc", FULL_TOKEN_PROBABILITY):
            _probability(getattr(self, name), name)

    def configuration(self) -> StrategyConfiguration:
        return StrategyConfiguration(DRA, {**asdict(self), "generations": 1, "moderation": "cache_only", "early_stop": "evaluator_success"})


def dra_profile(parameters: Mapping[str, object]) -> DRAProfile:
    defaults = DRAProfile()
    return DRAProfile(
        max_variants=cast(int, parameters.get("max_variants", defaults.max_variants)),
        seed=cast(int, parameters.get("seed", defaults.seed)),
        toxic_trunc=cast(float, parameters.get("toxic_trunc", defaults.toxic_trunc)),
        benign_trunc=cast(float, parameters.get("benign_trunc", defaults.benign_trunc)),
        benign_full_token_prob=cast(float, parameters.get("benign_full_token_prob", defaults.benign_full_token_prob)),
    )


def _probability(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a probability between 0 and 1")
    if name != FULL_TOKEN_PROBABILITY and value == 1:
        raise ValueError(f"{name} must be less than 1")
