"""Named finite DRA parameters."""

from typing import NamedTuple

DRABudget = NamedTuple(
    "DRABudget",
    [
        (
            "candidates", int
        ),
        (
            "seed", int
        )
    ]
)
