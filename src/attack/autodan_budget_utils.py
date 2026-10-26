"""Preflight bounds for the pinned native genetic operations."""

from attack.autodan_limits import AutoDANLimits

MAX_REFERENCES = 128
PAIR = 2


def validate(budget: AutoDANLimits) -> None:
    """Reject unsupported populations and invalid bounds before inference.

    Args:
        budget (AutoDANLimits): Explicit search limits to check.

    Raises:
        ValueError: If native GA cannot run within the requested budget.
    """
    if budget.steps < 1 or not PAIR <= budget.batch <= MAX_REFERENCES:
        raise ValueError("Invalid AutoDAN steps or reference population")
    if not 0 < budget.elites < budget.batch or (budget.batch - budget.elites) % PAIR:
        raise ValueError("Invalid AutoDAN elite or offspring count")
    if budget.points < 1 or budget.seed < 0 or not budget.target:
        raise ValueError("Invalid AutoDAN crossover, seed or completion")
    if not 0 <= budget.crossover <= 1 or not 0 <= budget.mutation <= 1:
        raise ValueError("Invalid AutoDAN probabilities")
