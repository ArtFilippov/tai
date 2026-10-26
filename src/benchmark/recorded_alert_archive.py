"""Write the series index and classify finished journals without model calls."""

import json
from pathlib import Path
from typing import Any, override

from attack.run_capture_contract import RunCaptureContract
from base.alert_taxonomy_contract import AlertTaxonomyContract
from benchmark.alert_archive import AlertArchive
from benchmark.alert_ports import AlertPorts

FAILURE = "failure"
INVALID = "invalid"
NO_RESULT = "no_result"
ASSESSMENT_ERROR = "assessment_error"
SUCCESSFUL = "successful"


class RecordedAlertArchive(AlertArchive):
    """Bind the saved-run format to injected journal creation and recovery."""

    def __init__(
        self, ports: AlertPorts, taxonomy: AlertTaxonomyContract,
    ) -> None:
        """Keep the independent journal ports and published taxonomy.

        Args:
            ports (AlertPorts): Journal construction and recovery.
            taxonomy (AlertTaxonomyContract): Published micro categories.
        """
        super().__init__()
        self.ports = ports
        self.taxonomy = taxonomy

    @override
    def save(self, path: Path, report: dict[str, Any]) -> None:
        """Write the latest series index.

        Args:
            path (Path): Summary destination.
            report (dict[str, Any]): Current index.
        """
        _ = path.write_text(
            json.dumps(
                report, ensure_ascii=False, indent=2
            ) + "\n", encoding="utf-8"
        )

    @override
    def record(self, path: Path, event: dict[str, object]) -> None:
        """Record a failure before the run itself could write a journal.

        Args:
            path (Path): Journal destination.
            event (dict[str, object]): Failure or interruption.
        """
        (
            self.ports.journal(path)
            .record(event)
        )

    @override
    def failure(self, path: Path) -> str:
        """Distinguish a failed assessment from an execution failure.

        Args:
            path (Path): Saved journal for the failed attempt.

        Returns:
            str: Invalid assessment or failed execution.
        """
        try:
            saved = (
                self.ports.view(path)
                .read()
            )
        except (ValueError, OSError):
            return FAILURE
        return INVALID if saved["status"] == FAILURE and saved["reason"] == ASSESSMENT_ERROR else FAILURE

    @override
    def outcome(self, path: Path) -> dict[str, str]:
        """Classify only the final, linked assessment of a finished journal.

        Args:
            path (Path): Saved journal of an attempt.

        Returns:
            dict[str, str]: Outcome with an optional error.
        """
        try:
            saved = (
                self.ports.view(path)
                .read()
            )
        except (ValueError, OSError) as exc:
            return {
                "outcome": INVALID,
                "error": repr(exc)
            }
        verdict = INVALID
        if saved["status"] == FAILURE:
            verdict = INVALID if saved["reason"] == ASSESSMENT_ERROR else FAILURE
        elif saved["status"] == NO_RESULT:
            verdict = NO_RESULT
        elif saved["status"] in {"completed", "success", "round_limit"} and saved["text"] is not None and saved["assessment"] in {True, False}:
            verdict = SUCCESSFUL if saved["assessment"] is True else "unsuccessful"
        return {"outcome": verdict}

    @override
    def categories(self) -> list[str]:
        """Return the full published microcategory order.

        Returns:
            list[str]: All canonical microcategories.
        """
        return self.taxonomy.categories()

    @override
    def capture(self) -> RunCaptureContract:
        """Start an independent exception boundary for one attempt.

        Returns:
            RunCaptureContract: Isolated attempt failure.
        """
        return self.ports.capture()
