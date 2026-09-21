"""External persistence of finished runs; never calls a model or an evaluator."""

import json
from dataclasses import asdict
from pathlib import Path

from core.primitives.diagnostics import DiagnosticRedactor

from .artifact_io import render_json, write_json, write_text
from .journal_ports import Journal
from .records import RunRecord
from .status import RunSnapshot, UserState
from .user import User

__all__ = ["RunRecord", "capture_run", "save_run", "write_json"]


def _validate_snapshot(snapshot: RunSnapshot) -> None:
    status, result = snapshot.status, snapshot.result
    if status.run_id is None or status.state not in {
        UserState.COMPLETED, UserState.STOPPED, UserState.FAILED,
    }:
        raise ValueError("Only a finished run can be saved")
    if status.result_id != (result.id if result is not None else None):
        raise ValueError("Snapshot result does not match its status")
    if status.evaluation is not None and (
        result is None or status.evaluation.result_id != result.id
        or not _is_boolean(status.evaluation.success)
    ):
        raise ValueError("Snapshot evaluation must be boolean and belong to its Message")


def _is_boolean(value: object) -> bool:
    return isinstance(value, bool)


def capture_run(
    user: User,
    *,
    journal: Journal | None = None,
    settings: object = None,
    final_settings: object = None,
    redactor: DiagnosticRedactor | None = None,
) -> RunRecord:
    """Capture after wait, using original User evidence and available settings.

    A retained journal supplies per-run diagnostic settings and redaction policy,
    never the result or terminal state. Bare Users can supply assembly settings
    explicitly. No private User state or participant is inspected.

    Args:
        user (User): User whose worker has finished.
        journal (Journal | None, default=None): Optional retained journal for this run.
        settings (object, default=None): Explicit initial assembly configuration.
        final_settings (object, default=None): Explicit final participant configuration.
        redactor (DiagnosticRedactor | None, default=None): Diagnostic policy for a bare User.

    Returns:
        RunRecord: Detached original evidence and redacted diagnostic metadata.

    Raises:
        ValueError: The supplied journal does not belong to the finished run.
    """
    snapshot = user.snapshot()
    status, result = snapshot.status, snapshot.result
    _validate_snapshot(snapshot)
    if journal is not None:
        if not journal.finished or journal.status().run_id != status.run_id:
            raise ValueError("Journal must belong to the finished snapshot's run")
        settings = journal.settings()
        final_settings = journal.final_settings()
        redactor = journal.redactor
    policy = redactor or DiagnosticRedactor()
    metadata = policy.snapshot(
        {
            **asdict(status),
            "result": json.loads(result.dump()) if result is not None else None,
            "settings": settings,
            "final_settings": final_settings,
        }
    )
    return RunRecord(snapshot, render_json(metadata))


def save_run(record: RunRecord, artifacts_dir: str | Path) -> Path:
    """Write final artifacts under <artifacts_dir>/<run_id>; errors propagate.

    This function has no live User reference. A failure or signal during storage
    cannot rewrite its outcome. Metadata is diagnostic; result.txt is verbatim.

    Args:
        record (RunRecord): Detached finished-run evidence.
        artifacts_dir (str | Path): Parent directory for run artifacts.

    Returns:
        Path: Directory containing the saved run artifacts.
    """
    snapshot = record.snapshot
    assert snapshot.status.run_id is not None
    directory = Path(artifacts_dir) / snapshot.status.run_id
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.txt"
    if snapshot.result is None:
        result_path.unlink(missing_ok=True)
    else:
        write_text(result_path, snapshot.result.to_string())
    write_text(directory / "metadata.json", record.dump_metadata())
    return directory
