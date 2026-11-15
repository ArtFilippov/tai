"""Shared DAN section options for the live entry points."""

from argparse import ArgumentParser
from collections.abc import Sequence


def options(parser: ArgumentParser) -> None:
    """Expose the same named-section selection in standalone and ALERT commands.

    Args:
        parser (ArgumentParser): Command parser receiving DAN options.
    """
    _ = parser.add_argument(
        "--dan-sections", nargs="+", choices=("named",), default=["named"],
        help="DAN prompt sections (currently only named; default: named)",
    )


def sections(selection: Sequence[str]) -> None:
    """Validate the section currently available at every public entry point.

    Args:
        selection (Sequence[str]): Explicit or default section names.

    Raises:
        ValueError: If selection is empty or contains unavailable sections.
    """
    if not selection or any(section != "named" for section in selection):
        raise ValueError("DAN sections must be nonempty; only 'named' is available")
