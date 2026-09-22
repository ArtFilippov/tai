"""SIGINT admission/finalization/persistence races and real CLI stop requests."""

import json
import signal
from pathlib import Path
from threading import Thread

import pytest
from qwen_fakes import LocalFixture
from series_interrupt_support import AdmissionFaults, InterruptRace, ReleaseGate
from series_support import MANIFEST, TAXONOMY, positive_responses, qwen_profile

from alert.report import reconstruct_report
from app.series import SeriesResult, run_alert_series

pytest_plugins = ("qwen_fakes",)


def _assert_saved_race(result: SeriesResult, boundary: str, attempts: int) -> None:
    assert result.exit_code == 130 and result.report["complete"] is False
    index = json.loads((result.directory / "index.json").read_text())
    assert index["interrupted"] is True and index["complete"] is False
    assert index["finished_at"] is not None and len(index["entries"]) == attempts
    assert sum(group["total"] for group in result.report["groups"].values()) == attempts
    assert reconstruct_report(result.directory / "index.json") == result.report
    assert json.loads((result.directory / "report.json").read_text()) == result.report
    for entry in index["entries"]:
        metadata = json.loads((result.directory / entry["artifacts"]["metadata.json"]).read_text())
        assert entry["state"] == metadata["state"]
        assert entry["reason"] == metadata["reason"]
        assert entry["evaluation"] == metadata["evaluation"]
    if boundary == "finalization":
        entry = index["entries"][0]
        assert entry["state"] == "STOPPED" and entry["reason"] == "user_stop"
        assert entry["evaluation"]["success"] is True
        assert entry["classification"]["evaluated"] is False
        assert result.report["groups"]["hate"]["success_percent"] is None
    if boundary in {"run_save", "first_save", "last_save"}:
        assert all(entry["state"] == "COMPLETED" for entry in index["entries"])


@pytest.mark.parametrize(("boundary", "attempts"), [
    ("construction", 0), ("initial_save", 0), ("before_start", 1), ("after_start", 1),
    ("finalization", 1), ("run_save", 1), ("first_save", 1), ("last_save", 32),
])
def test_interrupt_races_preserve_actual_outcomes_and_attempted_prefix(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str, attempts: int,
) -> None:
    lib, kwargs, _ = local
    profile = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    race = InterruptRace(boundary)
    race.install_lifecycle(monkeypatch)
    race.install_storage(monkeypatch)
    previous = signal.getsignal(signal.SIGINT)
    result = run_alert_series(MANIFEST, profile, tmp_path / "runs", taxonomy_path=TAXONOMY, jobs=1)
    assert race.raised.is_set()
    assert signal.getsignal(signal.SIGINT) is previous
    _assert_saved_race(result, boundary, attempts)


@pytest.mark.parametrize("phase", ["construction", "after_start"])
def test_creation_or_post_admission_error_is_saved_and_next_run_waits(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    lib, kwargs, _ = local
    profile = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    faults = AdmissionFaults(phase)
    faults.install(monkeypatch)
    result = run_alert_series(MANIFEST, profile, tmp_path / "runs", taxonomy_path=TAXONOMY, jobs=1)
    index = json.loads((result.directory / "index.json").read_text())
    assert result.exit_code == 1 and result.report["complete"] is True
    assert len(index["entries"]) == faults.constructed == 32
    failed = index["entries"][1]
    assert failed["error"]["phase"] == ("create_experiment" if phase == "construction" else "start")
    assert failed["classification"]["evaluated"] is False
    if phase == "construction":
        assert failed["run_id"] is None and failed["state"] is None
    else:
        assert failed["run_id"] == faults.users[1].status().run_id
        assert faults.users[1].wait(0)
    assert index["entries"][2]["state"] == "COMPLETED"


def test_terminal_status_before_worker_cleanup_does_not_admit_next_run(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib, kwargs, _ = local
    profile = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    gate = ReleaseGate()
    gate.install(monkeypatch)
    controller = Thread(target=gate.control)
    controller.start()
    try:
        result = run_alert_series(MANIFEST, profile, tmp_path / "runs", taxonomy_path=TAXONOMY, jobs=1)
    finally:
        gate.release.set()
        controller.join(10)
    assert gate.checked.is_set() and not result.exit_code
    assert len(gate.users) == 32
