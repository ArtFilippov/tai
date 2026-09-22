"""Deterministic overlap, out-of-order saving, bounded slots and multi-User stop."""

import json
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from parallel_series_support import ParallelScenario
from series_support import MANIFEST, TAXONOMY

from alert.accounting import SeriesReport, build_series_report
from alert.report import reconstruct_report
from alert.values import array_value, object_value
from app.series import SeriesResult, run_alert_series
from app.series_storage import SeriesFiles


@pytest.mark.parametrize("jobs", [1, 3, 32])
def test_bounded_overlap_reverse_completion_preserves_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: int,
) -> None:
    scenario = ParallelScenario()
    scenario.install(monkeypatch)

    def control() -> None:
        try:
            scenario.reverse_completion(jobs)
        finally:
            scenario.release_all()

    with ThreadPoolExecutor(max_workers=1) as pool:
        controlled = pool.submit(control)
        result = run_alert_series(MANIFEST, Path("examples/fitd-alert-qwen.toml"), tmp_path, taxonomy_path=TAXONOMY, jobs=jobs)
        controlled.result(5)
    assert not result.exit_code
    scenario.assert_independent()
    index = json.loads((result.directory / "index.json").read_text())
    assert len(index["entries"]) == 32 and index["effective_jobs"] == jobs
    assert reconstruct_report(result.directory / "index.json") == result.report
    assert_snapshots(scenario, jobs)
    assert_saved_identity(result)


def assert_snapshots(scenario: ParallelScenario, jobs: int) -> None:
    for snapshot in scenario.snapshots:
        entries = [object_value(entry, "entry") for entry in array_value(snapshot["entries"], "entries")]
        assert sum(entry["phase"] == "active" for entry in entries) <= jobs
        report = build_series_report(snapshot)
        assert sum(group["evaluated"] for group in report["groups"].values()) == sum(entry["phase"] == "finished" for entry in entries)


def assert_saved_identity(result: SeriesResult) -> None:
    index = json.loads((result.directory / "index.json").read_text())
    for i, entry in enumerate(index["entries"]):
        metadata = json.loads((result.directory / entry["artifacts"]["metadata.json"]).read_text())
        assert entry["sequence"] == i + 1
        assert metadata["settings"]["base"]["id"] == entry["base_id"]
        assert metadata["evaluation"] == entry["evaluation"]
        assert (result.directory / entry["artifacts"]["result.txt"]).read_text() == f"Accepted answer {i}"


def test_sigint_stops_all_active_users_and_preserves_completed_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = ParallelScenario()
    scenario.install(monkeypatch)

    def control() -> None:
        try:
            interrupt_active(scenario)
        finally:
            scenario.release_all()

    with ThreadPoolExecutor(max_workers=1) as pool:
        controlled = pool.submit(control)
        result = run_alert_series(MANIFEST, Path("examples/fitd-alert-qwen.toml"), tmp_path, taxonomy_path=TAXONOMY, jobs=3)
        controlled.result(5)
    assert result.exit_code == 130 and not result.report["complete"]
    index = json.loads((result.directory / "index.json").read_text())
    assert len(index["entries"]) == 3
    for entry in index["entries"]:
        assert entry["state"] == "STOPPED" and entry["phase"] == "finished"
        events = [json.loads(line) for line in (result.directory / entry["artifacts"]["events.jsonl"]).read_text().splitlines()]
        assert any(event["kind"] == "call_completed" and event["data"]["role"] == "Target" for event in events)
        assert any(event["kind"] == "resources_released" for event in events)
    assert reconstruct_report(result.directory / "index.json") == result.report


def interrupt_active(scenario: ParallelScenario) -> None:
    for event in scenario.entered[:3]:
        assert event.wait(5)
    os.kill(os.getpid(), signal.SIGINT)
    os.kill(os.getpid(), signal.SIGINT)
    assert scenario.stopped.wait(5)
    assert len(scenario.profiles) == 3


def test_shared_write_failure_drains_and_saves_every_active_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = ParallelScenario()
    scenario.install(monkeypatch)
    persist = SeriesFiles.persist
    failed = False

    def fail(store: SeriesFiles, index: dict[str, object], report: SeriesReport) -> None:
        nonlocal failed
        if len(scenario.profiles) == 3 and not failed:
            for entered in scenario.entered[:3]:
                assert entered.wait(5)
            failed = True
            scenario.release_all()
            raise OSError("shared write failed")
        persist(store, index, report)

    monkeypatch.setattr(SeriesFiles, "persist", fail)
    with pytest.raises(OSError, match="shared write failed"):
        run_alert_series(MANIFEST, Path("examples/fitd-alert-qwen.toml"), tmp_path, taxonomy_path=TAXONOMY, jobs=3)
    assert failed and len(scenario.profiles) == 3
    metadata = list(tmp_path.glob("*/runs/*/metadata.json"))
    assert len(metadata) == 3
    assert all(json.loads(path.read_text())["state"] in {"STOPPED", "COMPLETED"} for path in metadata)


@pytest.mark.parametrize("jobs", [0, -1, True])
def test_invalid_jobs_precedes_any_artifact_creation(tmp_path: Path, jobs: int) -> None:
    with pytest.raises(ValueError, match="jobs"):
        run_alert_series(MANIFEST, Path("examples/fitd-alert-vllm.toml"), tmp_path / "runs", taxonomy_path=TAXONOMY, jobs=jobs)
    assert not (tmp_path / "runs").exists()
