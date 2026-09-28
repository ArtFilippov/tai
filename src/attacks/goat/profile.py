"""Bounded GOAT conversation configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, cast

from attacks.shared.contracts import GOAT, StrategyConfiguration

PARAMETERS = frozenset({"max_calls_per_conv"})


class GOATOptions(TypedDict, total=False):
    max_calls_per_conv: int


@dataclass(frozen=True)
class GOATProfile:
    max_calls_per_conv: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.max_calls_per_conv, bool) or not isinstance(cast(object, self.max_calls_per_conv), int) or self.max_calls_per_conv <= 0:
            raise ValueError("max_calls_per_conv must be a positive integer")

    def configuration(self) -> StrategyConfiguration:
        return StrategyConfiguration(
            GOAT, {"max_calls_per_conv": self.max_calls_per_conv, "end_condition": "verify", "generations": 1},
            self.max_calls_per_conv,
        )


def goat_profile(parameters: Mapping[str, object]) -> GOATProfile:
    return GOATProfile(cast(int, parameters.get("max_calls_per_conv", 5)))
