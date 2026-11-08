"""Shared standalone and ALERT BEAST search arguments."""

from argparse import ArgumentParser


def options(parser: ArgumentParser) -> None:
    """Expose only garak's four search dimensions.

    Args:
        parser (ArgumentParser): Standalone or ALERT command.
    """
    _ = parser.add_argument(
        "--beast-beams", type=int, default=15
    )
    _ = parser.add_argument(
        "--beast-samples", type=int, default=15
    )
    _ = parser.add_argument(
        "--beast-suffix-len", type=int, default=40
    )
    _ = parser.add_argument(
        "--beast-trials", type=int, default=1
    )
