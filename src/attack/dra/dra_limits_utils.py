"""Validate a DRA run before model inference."""

from attack.dra.dra_limits import DRALimits

TEMPLATE_COUNT = 4


def validate(budget: DRALimits) -> None:
    """Reject unsupported template counts or negative seeds.

    Args:
        budget (DRALimits): Requested run bounds.

    Raises:
        ValueError: If the bounds are invalid.
    """
    if not 1 <= budget.candidates <= TEMPLATE_COUNT or budget.seed < 0:
        raise ValueError("Invalid DRA candidate count or seed")
