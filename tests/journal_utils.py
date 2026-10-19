"""Temporary journals shared by controlled test runs."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def temporary_journal(name: str) -> Generator[Path]:
    """Provide an isolated journal path for one check.

    Args:
        name (str): Journal file name.

    Yields:
        Path: Fresh path inside a temporary directory.
    """
    with TemporaryDirectory() as directory:
        yield Path(directory) / name
