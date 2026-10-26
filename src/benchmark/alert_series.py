"""A durable index of known outcomes for an ALERT attack series."""

import json
from pathlib import Path
from typing import Any, override

from benchmark.alert_series_contract import AlertSeriesContract

INTERRUPTED = "interrupted"


class AlertSeries(AlertSeriesContract):
    """Keep the attempt index current as the outer loop performs attacks."""

    def __init__(
        self, manifest: Path, directory: Path, planned: int,
    ) -> None:
        """Hold the output destination and the number of planned attempts.

        Args:
            manifest (Path): Original pool manifest.
            directory (Path): Reserved output directory.
            planned (int): Total attempts across all factories.
        """
        super().__init__()
        self.manifest = manifest
        self.directory = directory
        self.planned = planned
        self.entries: list[dict[str, object]] = []

    @override
    def report(self) -> dict[str, Any]:
        """Describe only known attempts, texts and execution outcomes.

        Returns:
            dict[str, Any]: Current plan and saved observations.
        """
        return {
            "manifest": str(self.manifest),
            "planned": self.planned,
            "attempted": len(self.entries),
            "complete": len(self.entries) == self.planned and all(
                entry["outcome"] != INTERRUPTED for entry in self.entries
            ),
            "runs": self.entries
        }

    @override
    def save(self) -> None:
        """Save the latest index in the launch's root directory."""
        _ = (self.directory / "summary.json").write_text(
            json.dumps(
                self.report(),
                ensure_ascii=False, indent=2
            ) + "\n", encoding="utf-8"
        )

    @override
    def record(self, entry: dict[str, object]) -> None:
        """Commit one observed attempt immediately.

        Args:
            entry (dict[str, object]): Available request, attack and outcome facts.
        """
        self.entries.append(entry)
        self.save()
