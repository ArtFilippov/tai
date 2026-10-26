"""Register DRA controls for ALERT without coupling command modules."""

from argparse import ArgumentParser


def options(parser: ArgumentParser) -> None:
    """Add DRA candidate and seed controls to the series command.

    Args:
        parser (ArgumentParser): ALERT argument parser.
    """
    _ = parser.add_argument(
        "--no-dra", action="store_true", help="omit DRA"
    )
    _ = parser.add_argument(
        "--dra-candidates", type=int, default=4
    )
    _ = parser.add_argument(
        "--dra-seed", type=int, default=0
    )
