"""Durable newline-delimited JSON event storage."""

import json
import os
from pathlib import Path
from typing import override

from fitd.journal import Journal


class DiskJournal(Journal):
    """A run's append-only file, independent of garak's global report."""

    def __init__(self, path: Path) -> None:
        """Keep the path for this run.

        Args:
            path (Path): A fresh journal path.
        """
        super().__init__()
        self.path = path

    @override
    def record(self, event: dict[str, object]) -> None:
        with self.path.open(
            "a", encoding="utf-8"
        ) as stream:
            _ = stream.write(
                json.dumps(
                    event, ensure_ascii=False
                ) + "\n"
            )
            stream.flush()
            os.fsync(
                stream.fileno()
            )
