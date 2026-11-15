"""Shared single-technique Smuggling selection at command boundaries."""

from argparse import ArgumentParser
from collections.abc import Sequence


def options(parser: ArgumentParser) -> None:
    """Expose the same technique selection in standalone and ALERT commands.

    Args:
        parser (ArgumentParser): Command parser receiving Smuggling options.
    """
    _ = parser.add_argument(
        "--smuggling-techniques", nargs="+", choices=("homoglyph",), default=["homoglyph"],
        help="Smuggling techniques (default: homoglyph; only homoglyph is available in this milestone)"
    )


def techniques(selection: Sequence[str]) -> None:
    """Reject empty and unsupported technique selections before model access.

    Args:
        selection (Sequence[str]): Explicit or default technique names.

    Raises:
        ValueError: If selection is empty or contains unsupported techniques.
    """
    if not selection or set(selection) != {"homoglyph"}:
        raise ValueError("Smuggling techniques must be a nonempty subset of homoglyph")
