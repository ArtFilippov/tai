"""File-backed series provenance and independent artifact inspection."""

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import override

from alert.accounting import SeriesReport
from core.runs import RunSnapshot
from core.runs.storage import write_json

from .series_contracts import SeriesStore

ARTIFACTS = ("result.txt", "events.jsonl", "metadata.json")


@dataclass(frozen=True)
class SeriesFiles(SeriesStore):
    _directory: Path
    inputs: dict[str, bytes]
    paths: dict[str, Path]
    settings: Callable[[], dict[str, object]]
    redact: Callable[[str], str]
    initial_settings: dict[str, object]

    @property
    @override
    def directory(self) -> Path:
        return self._directory

    @override
    def initialize(self) -> dict[str, object]:
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "runs").mkdir()
        _ = (self.directory / "manifest.json").write_bytes(self.inputs["manifest"])
        _ = (self.directory / "ALERT_taxonomy.md").write_bytes(self.inputs["taxonomy"])
        write_json(self.directory / "effective-profile.json", self.initial_settings)
        return {
            "schema_version": 1, "series_id": self.directory.name, "manifest": "manifest.json",
            "effective_profile": "effective-profile.json", "final_settings": "final-settings.json",
            "sources": {name: {"path": self.redact(str(path.resolve())),
                               "sha256": hashlib.sha256(self.inputs[name]).hexdigest()}
                        for name, path in self.paths.items()},
        }

    @override
    def persist(self, index: Mapping[str, object], report: SeriesReport) -> None:
        write_json(self.directory / "index.json", index)
        write_json(self.directory / "report.json", report)

    @override
    def finish(self) -> None:
        write_json(self.directory / "final-settings.json", self.settings())

    def _artifact(self, relative: Path) -> str | None:
        return str(relative) if (self.directory / relative).is_file() else None

    @override
    def inspect(self, snapshot: RunSnapshot | None) -> dict[str, object]:
        run_id = snapshot.status.run_id if snapshot is not None else None
        run_dir = Path("runs") / run_id if run_id is not None else None
        artifacts: dict[str, str | None] = {"directory": str(run_dir) if run_dir is not None else None}
        errors: dict[str, object] = {}
        for name in ARTIFACTS:
            artifacts[name] = None
            if run_dir is not None:
                try:
                    artifacts[name] = self._artifact(run_dir / name)
                except OSError as exc:
                    # Inspection is run-local. Preserve the actual terminal
                    # snapshot and inspect remaining links after one failed stat.
                    errors[name] = {"type": type(exc).__name__, "message": self.redact(str(exc))}
        return {"artifacts": artifacts, "artifact_errors": errors}
