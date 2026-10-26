"""Named parameters for bounded native GA with an explicit completion."""

from typing import NamedTuple

AutoDANBudget = NamedTuple(
    "AutoDANBudget",
    [
        (
            "steps", int
        ),
        (
            "batch", int
        ),
        (
            "elites", int
        ),
        (
            "points", int
        ),
        (
            "crossover", float
        ),
        (
            "mutation", float
        ),
        (
            "seed", int
        ),
        (
            "target", str
        )
    ]
)
