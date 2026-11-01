"""Register Bad Characters deletion controls for ALERT."""

from argparse import ArgumentParser


def options(parser: ArgumentParser) -> None:
    """Add deletion limits and variant selection to the series command.

    Args:
        parser (ArgumentParser): ALERT argument parser.
    """
    _ = parser.add_argument(
        "--no-deletion", action="store_true", help="omit Bad Characters deletion"
    )
    _ = parser.add_argument(
        "--deletion-perturbations", type=int, default=1
    )
    _ = parser.add_argument(
        "--deletion-positions", type=int, default=8
    )
    _ = parser.add_argument(
        "--deletion-ascii", type=int, default=4
    )
