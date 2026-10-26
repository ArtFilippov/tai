"""Explicit offline preparation of the pinned ALERT manifest."""

import argparse
from argparse import Namespace
from pathlib import Path

from base.alert_manifest import AlertManifest


def main() -> None:
    """Validate the official source snapshot and write the manifest."""
    parser = argparse.ArgumentParser(description="Build an offline ALERT pool from pinned alert.jsonl")
    _ = parser.add_argument(
        "source", type=Path
    )
    _ = parser.add_argument(
        "manifest", type=Path
    )
    launch(
        parser.parse_args()
    )


def launch(arguments: Namespace) -> None:
    """Save the prepared manifest selected on the command line.

    Args:
        arguments (Namespace): Source and output paths.
    """
    AlertManifest(arguments.source).save(arguments.manifest)


if __name__ == "__main__":
    main()
