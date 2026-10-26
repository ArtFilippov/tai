"""Ordered independent runs with a summary derived from saved outcomes."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, override

from attack.alert_archive import AlertArchive
from attack.alert_run import AlertRun
from attack.alert_series_contract import AlertSeriesContract
from attack.run_capture_contract import RunCaptureContract
from base.alert_identity import AlertIdentity
from base.base_request import BaseRequest

INTERRUPTED = "interrupted"
SUCCESSFUL = "successful"
FAILURE = "failure"
INVALID = "invalid"
NO_RESULT = "no_result"
ERROR = "error"


class AlertSeries(AlertSeriesContract):
    """Own a new output directory and preserve each attempted run's outcome."""

    def __init__(
        self, bases: Sequence[AlertIdentity], taxonomy: Mapping[str, object],
        directory: Path, manifest: Path, *, archive: AlertArchive,
    ) -> None:
        """Keep the plan, taxonomy and destination for the series.

        Args:
            bases (Sequence[AlertIdentity]): Selected original requests.
            taxonomy (Mapping[str, object]): Macro categories in canonical order.
            directory (Path): New journal directory.
            manifest (Path): Source manifest location.
            archive (AlertArchive): Persistence and classification boundary.
        """
        super().__init__()
        self.bases = bases
        self.taxonomy = taxonomy
        self.directory = directory
        self.manifest = manifest
        self.archive = archive
        self.entries: list[dict[str, str | int]] = []

    @override
    def report(self) -> dict[str, Any]:
        """Count only verified terminal assessments, without calling a model.

        Returns:
            dict[str, Any]: Counts and linked journal outcomes.
        """
        return {
            "manifest": str(self.manifest),
            "complete": len(self.entries) == len(self.bases) and all(
                entry["outcome"] != INTERRUPTED for entry in self.entries
            ),
            "planned": len(self.bases),
            "attempted": len(self.entries),
            "runs": self.entries,
            "micro": {category: self.count(category) for category in self.archive.categories()},
            "macro": {
                macro: self.count(macro, key="macro") for macro in self.taxonomy
            },
        }

    @override
    def count(self, category: str, *, key: str = "microcategory") -> dict[str, object]:
        """Aggregate recorded outcomes for one canonical category or group.

        Args:
            category (str): Category to count.
            key (str, default='microcategory'): Entry field naming the category.

        Returns:
            dict[str, object]: Outcome counts and percentage.
        """
        items = [entry for entry in self.entries if entry[key] == category]
        planned = sum(base.metadata()[key] == category for base in self.bases)
        evaluated = sum(entry["outcome"] in {SUCCESSFUL, "unsuccessful"} for entry in items)
        successful = sum(entry["outcome"] == SUCCESSFUL for entry in items)
        return {
            "planned": planned,
            "attempted": len(items),
            "evaluated": evaluated,
            "successful": successful,
            "unsuccessful": evaluated - successful,
            "unevaluated": len(items) - evaluated,
            "failure": sum(entry["outcome"] == FAILURE for entry in items),
            "no_result": sum(entry["outcome"] == NO_RESULT for entry in items),
            "invalid": sum(entry["outcome"] == INVALID for entry in items),
            "rate": 100 * successful / evaluated if evaluated else None,
        }

    @override
    def save(self) -> None:
        """Replace only this series' own summary after every completed attempt."""
        self.archive.save(
            self.directory / "summary.json",
            self.report()
        )

    @override
    def invoke(self, runs: AlertRun, base: BaseRequest, path: Path, *, attempt: RunCaptureContract) -> None:
        """Run one attempt inside its independent failure boundary.

        Args:
            runs (AlertRun): Shared per-attempt execution.
            base (BaseRequest): Selected original request.
            path (Path): Reserved journal path.
            attempt (RunCaptureContract): Current exception boundary.
        """
        with attempt.guard():
            runs.run(
                base, path
            )

    @override
    def run(self, runs: AlertRun) -> None:
        """Attempt every base in order, retaining errors and partial interruptions.

        Args:
            runs (AlertRun): Fresh isolated run for each base.

        Raises:
            KeyboardInterrupt: When the series has been interrupted and saved.
        """
        for index, base in enumerate(
            self.bases, start=1
        ):
            path = self.directory.joinpath(f"{index:03d}-{base.identity}.jsonl")
            path.touch(exist_ok=False)
            entry: dict[str, str | int] = {
                "id": base.identity,
                "microcategory": base.microcategory,
                "macro": base.macro,
                "journal": path.name,
                "outcome": "pending",
            }
            attempt = self.archive.capture()
            try:
                self.invoke(
                    runs, base, path, attempt=attempt
                )
            except KeyboardInterrupt:
                if not path.stat().st_size:
                    self.archive.record(
                        path,
                        {
                            "event": "interruption",
                            "base_metadata": dict(
                                base.metadata()
                            )
                        }
                    )
                entry["outcome"] = INTERRUPTED
                self.entries.append(entry)
                self.save()
                raise
            if attempt.failed():
                exc = attempt.error()
                if not path.stat().st_size:
                    self.archive.record(
                        path,
                        {
                            "event": "startup_error",
                            "base_metadata": dict(
                                base.metadata()
                            ),
                            "error": exc
                        }
                    )
                entry.update(
                    {
                        "outcome": self.archive.failure(path),
                        "error": exc
                    }
                )
                self.entries.append(entry)
            else:
                verdict = self.archive.outcome(path)
                entry["outcome"] = verdict["outcome"]
                if ERROR in verdict:
                    entry[ERROR] = verdict[ERROR]
                self.entries.append(entry)
            self.save()
