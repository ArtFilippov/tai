"""Shared DAN section options for the live entry points."""

from argparse import ArgumentParser
from collections.abc import Sequence


def options(parser: ArgumentParser) -> None:
    """Expose the same section selection in standalone and ALERT commands.

    Args:
        parser (ArgumentParser): Command parser receiving DAN options.
    """
    _ = parser.add_argument(
        "--dan-sections", nargs="+", choices=("named", "ablation", "wild"), default=["named", "ablation", "wild"],
        help="DAN prompt sections (default: named ablation wild; canonical traversal order)",
    )


def sections(selection: Sequence[str]) -> None:
    """Validate the selected sections at every public entry point.

    Args:
        selection (Sequence[str]): Explicit or default section names.

    Raises:
        ValueError: If selection is empty or contains unavailable sections.
    """
    if not selection or set(selection) - {"named", "ablation", "wild"}:
        raise ValueError("DAN sections must be a nonempty subset of named, ablation, wild")
