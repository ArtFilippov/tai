"""Shared positive Phrasing generation budget at command boundaries."""

from argparse import ArgumentParser


def options(parser: ArgumentParser) -> None:
    """Expose the same Attacker-call budget in standalone and ALERT commands.

    Args:
        parser (ArgumentParser): Command receiving Phrasing options.
    """
    _ = parser.add_argument(
        "--phrasing-attempts", type=int, default=20, help="Maximum Attacker calls per tense (default: 20)"
    )


def validate(tense: str, attempts: int) -> None:
    """Reject invalid tense identities and unbounded or empty generation budgets.

    Args:
        tense (str): Past or future identity of one independent run.
        attempts (int): Maximum Attacker calls.

    Raises:
        ValueError: If the tense or budget is invalid.
    """
    if tense not in ("past", "future") or attempts < 1:
        raise ValueError("Phrasing requires past/future and a positive generation budget")
