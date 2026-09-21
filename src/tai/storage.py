"""External persistence of finished runs; never calls a model or an evaluator."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tai._diagnostics import DiagnosticObject, DiagnosticRedactor
from tai.journal import RunJournal
from tai.users import RunSnapshot, User, UserState


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Detached evidence ready to save, independent of restarts and output paths."""

    snapshot: RunSnapshot
    _metadata_json: str = field(repr=False)

    @property
    def metadata(self) -> DiagnosticObject:
        return json.loads(self._metadata_json)


def capture_run(
    user: User,
    *,
    journal: RunJournal | None = None,
    settings: Any = None,
    final_settings: Any = None,
    redactor: DiagnosticRedactor | None = None,
) -> RunRecord:
    """Capture after wait, using original User evidence and available settings.

    A retained journal supplies per-run diagnostic settings and redaction policy,
    never the result or terminal state. Bare Users can supply assembly settings
    explicitly. No private User state or participant is inspected.
    """
    snapshot = user.snapshot()
    status, result = snapshot.status, snapshot.result
    if status.run_id is None or status.state not in (
        UserState.COMPLETED,
        UserState.STOPPED,
        UserState.FAILED,
    ):
        raise ValueError("Only a finished run can be saved")
    if status.result_id != (result.id if result is not None else None):
        raise ValueError("Snapshot result does not match its status")
    if status.evaluation is not None and (
        result is None
        or status.evaluation.result_id != result.id
        or type(status.evaluation.success) is not bool
    ):
        raise ValueError(
            "Snapshot evaluation must be boolean and belong to its Message"
        )
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
    return RunRecord(snapshot, _json(metadata))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _write_text(path: Path, text: str) -> None:
    """Atomically replace one file; propagate write/replace/cleanup failures."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, _json(value))


def save_run(record: RunRecord, artifacts_dir: str | Path) -> Path:
    """Write final artifacts under <artifacts_dir>/<run_id>; errors propagate.

    This function has no live User reference. A failure or signal during storage
    cannot rewrite its outcome. Metadata is diagnostic; result.txt is verbatim.
    """
    snapshot = record.snapshot
    assert snapshot.status.run_id is not None
    directory = Path(artifacts_dir) / snapshot.status.run_id
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.txt"
    if snapshot.result is None:
        result_path.unlink(missing_ok=True)
    else:
        _write_text(result_path, snapshot.result.to_string())
    _write_text(directory / "metadata.json", record._metadata_json)
    return directory
