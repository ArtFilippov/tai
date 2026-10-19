"""Durable newline-delimited JSON event storage."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import override

from fitd.journal import Journal


@dataclass(eq=False)
class DiskJournal(Journal):
    """A run's append-only file, independent of garak's global report.

    Attributes:
        path (Path): A fresh journal path.
    """

    path: Path

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
