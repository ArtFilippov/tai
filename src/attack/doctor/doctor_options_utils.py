"""Shared Doctor selection and seed at command boundaries."""

from argparse import ArgumentParser
from collections.abc import Sequence

TECHNIQUES = ("puppetry", "bypass", "bypass_leet")


def options(parser: ArgumentParser) -> None:
    """Expose Doctor settings in standalone and ALERT commands.

    Args:
        parser (ArgumentParser): Command parser receiving Doctor options.
    """
    _ = parser.add_argument(
        "--doctor-techniques", nargs="+", choices=TECHNIQUES, default=list(TECHNIQUES),
        help="Doctor techniques (execution order: puppetry, bypass, bypass_leet)"
    )
    _ = parser.add_argument(
        "--doctor-seed", type=int, default=0, help="Run-local leetspeak seed"
    )


def selection(techniques: Sequence[str]) -> Sequence[str]:
    """Validate and canonicalize a nonempty technique selection.

    Args:
        techniques (Sequence[str]): Requested technique names.

    Returns:
        Sequence[str]: Deduplicated names in canonical order.

    Raises:
        ValueError: If selection is empty or contains unsupported techniques.
    """
    if not techniques or not set(techniques) <= set(TECHNIQUES):
        raise ValueError("Doctor techniques must be a nonempty subset of puppetry, bypass, bypass_leet")
    return tuple(name for name in TECHNIQUES if name in techniques)
