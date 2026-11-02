"""Open wn's explicitly installed local database at the composition boundary."""

from pathlib import Path

import wn
from wn._config import config


def open_wordnet(directory: Path) -> wn.Wordnet:
    """Open an existing WordNet database without downloading anything.

    Args:
        directory (Path): Directory prepared outside inference.

    Returns:
        wn.Wordnet: Selected Open English WordNet 2023.

    Raises:
        ValueError: If the database or lexicon is missing.
    """
    if not (directory / "wn.db").is_file():
        raise ValueError(f"WordNet database is not prepared in {directory}")
    config.data_directory = directory
    if not wn.lexicons(lexicon="oewn:2023"):
        raise ValueError(f"WordNet lexicon oewn:2023 is not prepared in {directory}")
    return wn.Wordnet("oewn:2023")
