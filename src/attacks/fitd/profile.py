"""The initial FITD profile shared by execution, observation, and artifacts."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _FITDProfile:
    generations: int = 1
    max_calls_per_conv: int = 10
    num_level_queries: int = 3


FITD_PROFILE = _FITDProfile()
