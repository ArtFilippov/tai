"""Validated deletion-only parameters; generator caps select a native subset."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TypedDict

from attacks.shared.contracts import BADCHARS_DELETION, StrategyConfiguration

PARAMETERS = frozenset({"perturbation_budget", "max_position_candidates", "max_ascii_variants"})
ASCII_VARIANTS = 95


class DeletionOptions(TypedDict, total=False):
    perturbation_budget: int
    max_position_candidates: int
    max_ascii_variants: int


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class DeletionProfile:
    perturbation_budget: int = 1
    max_position_candidates: int = 8
    max_ascii_variants: int = 4

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _ = _positive_integer(value, name)
        if self.max_ascii_variants > ASCII_VARIANTS:
            raise ValueError("max_ascii_variants must not exceed 95 printable ASCII characters")

    def configuration(self) -> StrategyConfiguration:
        return StrategyConfiguration(BADCHARS_DELETION, {
            **asdict(self), "category": "deletion", "generations": 1,
            "follow_prompt_cap": False, "early_stop": "evaluator_success",
        })


def deletion_profile(parameters: Mapping[str, object]) -> DeletionProfile:
    defaults = DeletionProfile()
    return DeletionProfile(**{
        name: _positive_integer(parameters.get(name, default), name)
        for name, default in asdict(defaults).items()
    })
