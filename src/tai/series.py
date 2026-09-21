"""Sequential ALERT experiments using FITDUser's public lifecycle on the main thread."""

import hashlib
import platform
import signal
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from types import FrameType
from typing import Any
from uuid import uuid4

from tai import FITDExperiment, Message
from tai._config import RoleProfile, load_profile
from tai._diagnostics import DiagnosticRedactor
from tai._profile import FITD_PROFILE
from tai.alert import load_alert_manifest
from tai.series_report import _write_json, build_series_report, classify_outcome
from tai.storage import capture_run, save_run
from tai.users import RunSnapshot


@dataclass(frozen=True)
class SeriesResult:
    directory: Path
    report: dict[str, Any]
    exit_code: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _settings(profile: RoleProfile) -> dict[str, Any]:
    from tai._fitd import GARAK_REVISION

    return {
        "tai_version": version("tai"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "garak_revision": GARAK_REVISION,
        "fitd": asdict(
            replace(FITD_PROFILE, max_calls_per_conv=profile.max_calls_per_conv)
        ),
        **profile.settings(),
    }


def _capture(
    snapshot: RunSnapshot | None, directory: Path, redactor: DiagnosticRedactor
) -> dict[str, Any]:
    status = snapshot.status if snapshot is not None else None
    result = snapshot.result if snapshot is not None else None
    run_id = status.run_id if status is not None else None
    run_dir = Path("runs") / run_id if run_id is not None else None
    artifacts = {"directory": str(run_dir) if run_dir is not None else None}
    artifact_errors = {}
    for name in ("result.txt", "events.jsonl", "metadata.json"):
        artifacts[name] = None
        if run_dir is not None:
            try:
                if (directory / run_dir / name).is_file():
                    artifacts[name] = str(run_dir / name)
            except OSError as exc:
                # Inspection is run-local: retain the terminal snapshot and
                # inspect the other links even if this artifact is unavailable.
                artifact_errors[name] = {
                    "type": type(exc).__name__,
                    "message": redactor.text(str(exc)),
                }
    return {
        "state": status.state.value if status is not None else None,
        "reason": status.reason if status is not None else None,
        "run_id": run_id,
        "progress": asdict(status.progress) if status is not None else None,
        "result_id": status.result_id if status is not None else None,
        "result": {"kind": "Message", "id": result.id}
        if isinstance(result, Message)
        else None,
        "evaluation": asdict(status.evaluation)
        if status is not None and status.evaluation is not None
        else None,
        "artifacts": artifacts,
        "artifact_errors": artifact_errors,
    }


def run_alert_series(
    manifest_path: Path,
    profile_path: Path,
    artifacts_dir: Path,
    *,
    taxonomy_path: Path,
) -> SeriesResult:
    """Validate all inputs, then run one FITD per saved record in order.

    Main thread only (owns SIGINT temporarily). A single loaded RoleProfile holds
    shared Qwen weights. Each User supplies its base to the same evaluator contract.
    Configuration/series I/O errors raise; run-local errors are saved and continued.
    """
    paths = {
        "manifest": manifest_path,
        "profile": profile_path,
        "taxonomy": taxonomy_path,
    }
    inputs = {name: path.read_bytes() for name, path in paths.items()}
    manifest = load_alert_manifest(manifest_path, taxonomy_path=taxonomy_path)
    profile = load_profile(profile_path, alert_series=True)
    if any(path.read_bytes() != inputs[name] for name, path in paths.items()):
        raise ValueError("Series input files changed during validation")
    settings = _settings(profile)  # Settings only: no readiness/model calls.
    redactor = profile.redactor()
    sources = {
        name: {
            "path": redactor.text(str(path.resolve())),
            "sha256": hashlib.sha256(inputs[name]).hexdigest(),
        }
        for name, path in paths.items()
    }
    interrupted = False
    accepting_interrupts = True

    def interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        # Never raise through start/join, take a User lock, or perform I/O here.
        if accepting_interrupts:
            interrupted = True

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    current: FITDExperiment | None = None
    try:
        series_id = str(uuid4())
        directory = artifacts_dir / series_id
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "runs").mkdir()
        (directory / "manifest.json").write_bytes(inputs["manifest"])
        (directory / "ALERT_taxonomy.md").write_bytes(inputs["taxonomy"])
        _write_json(directory / "effective-profile.json", settings)
        began = monotonic()
        index: dict[str, Any] = {
            "schema_version": 1,
            "series_id": series_id,
            "manifest": "manifest.json",
            "effective_profile": "effective-profile.json",
            "final_settings": "final-settings.json",
            "sources": sources,
            "started_at": _now(),
            "finished_at": None,
            "duration_seconds": None,
            "interrupted": False,
            "complete": False,
            "planned": [
                {
                    "id": record.id,
                    "source_category": record.source_category,
                    "micro_category": record.micro_category,
                    "macro_category": record.macro_category,
                }
                for record in manifest.records
            ],
            "entries": [],
        }

        def save() -> dict[str, Any]:
            # SIGINT during either atomic write invalidates this snapshot. Repeat
            # once with interrupted=true, including at the final-record boundary.
            while True:
                index["interrupted"] = interrupted
                report = build_series_report(index)
                index["complete"] = report["complete"]
                _write_json(directory / "index.json", index)
                _write_json(directory / "report.json", report)
                if index["interrupted"] == interrupted:
                    return report

        save()  # Also validates artifact writes before the first start.
        for sequence, record in enumerate(manifest.records, 1):
            if interrupted:
                break
            current = None
            base = Message(record.base_request)
            started_at, start_time = _now(), monotonic()
            error = None
            phase = "create_experiment"
            try:
                current = profile.create_experiment(base, directory / "runs")
                # Interrupt during construction has not attempted start. Do not
                # count this or any later unstarted record in the denominator.
                if interrupted:
                    break
                phase = "start"
                current.start()
                phase = "wait"
                while True:
                    if interrupted:
                        current.stop()
                    if current.wait(0.05):
                        break
            except Exception as exc:
                error = {
                    "phase": phase,
                    "type": type(exc).__name__,
                    "message": redactor.text(str(exc)),
                }
                # A wrapper can raise after start admitted a worker. Join even
                # then; a visible terminal status alone does not release chats.
                if current is not None and current.status().run_id is not None:
                    current.stop()
                    current.wait()
            snapshot = None
            storage_error = None
            if current is not None:
                if current.status().run_id is None:
                    snapshot = RunSnapshot(current.status(), None)
                else:
                    snapshot = current.snapshot()
                    try:
                        record_snapshot = capture_run(current, journal=current.journal)
                        save_run(record_snapshot, directory / "runs")
                    except Exception as exc:
                        storage_error = {
                            "phase": "save",
                            "type": type(exc).__name__,
                            "message": redactor.text(str(exc)),
                        }
            outcome = _capture(snapshot, directory, redactor)
            if (
                error is None
                and current is not None
                and current.journal.error is not None
            ):
                failure = current.journal.error
                error = {
                    "phase": "journal",
                    "type": type(failure).__name__,
                    "message": redactor.text(str(failure)),
                }
            outcome["reason"] = (
                redactor.text(outcome["reason"])
                if outcome["reason"] is not None
                else None
            )
            entry = {
                **index["planned"][sequence - 1],
                "sequence": sequence,
                "base_id": base.id,
                "started_at": started_at,
                "finished_at": _now(),
                "duration_seconds": monotonic() - start_time,
                **outcome,
                "error": error,
                "storage_error": storage_error,
            }
            entry["classification"] = classify_outcome(entry)
            index["entries"].append(entry)
            current = None  # wait() has covered finalization and ownership release.
            save()
        _write_json(directory / "final-settings.json", _settings(profile))
        index["finished_at"] = _now()
        index["duration_seconds"] = monotonic() - began
        report = save()
        # Seal the terminal boundary only after the final files exist. A signal
        # between save's last check and this assignment still needs persistence;
        # later signals cannot turn exit_code into 130 with a complete old index.
        accepting_interrupts = False
        if index["interrupted"] != interrupted:
            report = save()
        failed = any(
            entry["error"] is not None
            or entry["storage_error"] is not None
            or entry["artifact_errors"]
            or entry["state"] != "COMPLETED"
            for entry in index["entries"]
        )
        return SeriesResult(directory, report, 130 if interrupted else int(failed))
    finally:
        try:
            if current is not None and current.status().run_id is not None:
                current.stop()
                current.wait()
        finally:
            signal.signal(signal.SIGINT, previous_handler)
