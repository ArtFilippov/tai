"""Run-local artifact failures and fatal shared-series storage failures."""

import errno
import json
import signal
from pathlib import Path

import pytest
from qwen_fakes import LocalFixture
from series_support import MANIFEST, TAXONOMY, positive_responses, qwen_profile

from alert import load_alert_manifest
from alert.report import reconstruct_report
from app import series_storage
from app.cli import main
from app.configuration import RoleProfile
from app.series import SeriesResult, run_alert_series
from tai import FITDExperiment, Message

pytest_plugins = ("qwen_fakes",)


def _install_storage_fault(
    monkeypatch: pytest.MonkeyPatch, users: list[FITDExperiment], filename: str, interrupt: bool,
) -> None:
    factory, write = RoleProfile.create_experiment, Path.write_text

    def create(profile: RoleProfile, base: Message, directory: Path) -> FITDExperiment:
        if users:
            index = json.loads((directory.parent / "index.json").read_text())
            assert index["entries"][0]["storage_error"]["phase"] == "save"
            assert users[-1].wait(0)
        user = factory(profile, base, directory)
        users.append(user)
        return user

    def persist(path: Path, data: str, encoding: str | None = None, errors: str | None = None, newline: str | None = None) -> int:
        if users and path.parent.name == users[0].status().run_id and path.name == filename:
            assert users[0].wait(0)
            assert users[0].status().state == "COMPLETED"
            if interrupt:
                signal.raise_signal(signal.SIGINT)
            raise OSError("cannot write final artifact")
        return write(path, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(RoleProfile, "create_experiment", create)
    monkeypatch.setattr(Path, "write_text", persist)


@pytest.mark.parametrize(("filename", "interrupt"), [
    ("result.txt.tmp", False), ("metadata.json.tmp", False), ("metadata.json.tmp", True),
])
def test_run_storage_failure_is_explicit_and_preserves_actual_completed_outcome(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str, interrupt: bool,
) -> None:
    lib, kwargs, _ = local
    profile = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    users: list[FITDExperiment] = []
    _install_storage_fault(monkeypatch, users, filename, interrupt)
    result = run_alert_series(MANIFEST, profile, tmp_path / "runs", taxonomy_path=TAXONOMY, jobs=1)
    assert result.exit_code == (130 if interrupt else 1)
    index = json.loads((result.directory / "index.json").read_text())
    first = index["entries"][0]
    assert first["state"] == "COMPLETED" and first["error"] is None
    assert first["storage_error"] == {"phase": "save", "type": "OSError", "message": "cannot write final artifact"}
    assert first["evaluation"]["success"] is True
    assert first["classification"] == {"evaluated": False, "success": None, "reason": "storage_error"}
    assert all(user.status().state == "COMPLETED" and user.wait(0) for user in users)
    assert len(users) == len(index["entries"]) == (1 if interrupt else 32)
    assert index["interrupted"] is interrupt
    assert result.report["complete"] is (not interrupt)
    assert sum(group["evaluated"] for group in result.report["groups"].values()) == len(users) - 1
    assert reconstruct_report(result.directory / "index.json") == result.report


@pytest.mark.parametrize("filename", ["index.json", "report.json", "final-settings.json"])
def test_shared_series_write_failure_terminates_command_after_join(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], filename: str,
) -> None:
    lib, kwargs, _ = local
    profile = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    users: list[FITDExperiment] = []
    factory, write = RoleProfile.create_experiment, series_storage.write_json

    def create(profile: RoleProfile, base: Message, directory: Path) -> FITDExperiment:
        user = factory(profile, base, directory)
        users.append(user)
        return user

    def persist(path: Path, value: object) -> None:
        if users and path.name == filename and users[-1].wait(0):
            assert users[-1].wait(0)
            raise OSError("shared series storage unavailable")
        write(path, value)

    monkeypatch.setattr(RoleProfile, "create_experiment", create)
    monkeypatch.setattr(series_storage, "write_json", persist)
    assert main(["alert-series", "--jobs", "1", "--manifest", str(MANIFEST), "--profile", str(profile),
                 "--artifacts-dir", str(tmp_path / "runs"), "--taxonomy", str(TAXONOMY)]) == 2
    assert "shared series storage unavailable" in capsys.readouterr().err
    assert len(users) == (32 if filename == "final-settings.json" else 1)
    assert all(user.wait(0) for user in users)


def _install_inspection_fault(
    monkeypatch: pytest.MonkeyPatch, users: list[FITDExperiment], unavailable: tuple[str, ...], interrupt: bool,
) -> list[str]:
    manifest = load_alert_manifest(MANIFEST, taxonomy_path=TAXONOMY)
    factory, is_file = RoleProfile.create_experiment, Path.is_file
    failed_stats: list[str] = []

    def create(profile: RoleProfile, base: Message, directory: Path) -> FITDExperiment:
        assert base.text == manifest.records[len(users)].base_request
        if users:
            assert users[-1].wait(0), "next attempt overlapped finalization"
            saved = json.loads((directory.parent / "index.json").read_text())
            assert len(saved["entries"]) == len(users)
            assert saved["entries"][0]["run_id"] == users[0].status().run_id
        user = factory(profile, base, directory)
        users.append(user)
        return user

    def inspect(path: Path) -> bool:
        if path.name in unavailable and users and path.parent.name == users[0].status().run_id:
            assert users[0].wait(0), "fault must occur after the worker is joined"
            assert users[0].status().state == "COMPLETED"
            failed_stats.append(path.name)
            if interrupt:
                signal.raise_signal(signal.SIGINT)
            raise OSError(errno.EIO, "artifact stat failed")
        return is_file(path)

    monkeypatch.setattr(RoleProfile, "create_experiment", create)
    monkeypatch.setattr(Path, "is_file", inspect)
    return failed_stats


def _assert_inspected_runs(result: SeriesResult, users: list[FITDExperiment], interrupt: bool) -> None:
    manifest = load_alert_manifest(MANIFEST, taxonomy_path=TAXONOMY)
    index = json.loads((result.directory / "index.json").read_text())
    expected_attempts = 1 if interrupt else 32
    assert len(index["entries"]) == len(users) == expected_attempts
    assert [entry["id"] for entry in index["entries"]] == [record.id for record in manifest.records[:expected_attempts]]
    assert all(user.wait(0) for user in users)
    for entry in index["entries"]:
        run_dir = result.directory / "runs" / entry["run_id"]
        metadata = json.loads((run_dir / "metadata.json").read_text())
        for key in ("state", "reason", "run_id", "progress", "result_id", "evaluation"):
            assert entry[key] == metadata[key]
        assert entry["state"] == "COMPLETED" and entry["error"] is None
        assert entry["result"] == {"kind": "Message", "id": metadata["result"]["id"]}
        assert entry["evaluation"]["success"] is True


def _assert_inspection_errors(result: SeriesResult, unavailable: tuple[str, ...], interrupt: bool) -> None:
    index = json.loads((result.directory / "index.json").read_text())
    affected = index["entries"][0]
    assert affected["artifact_errors"] == {
        name: {"type": "OSError", "message": "[Errno 5] artifact stat failed"} for name in unavailable
    }
    for name in ("result.txt", "events.jsonl", "metadata.json"):
        assert affected["artifacts"][name] == (None if name in unavailable else f"runs/{affected['run_id']}/{name}")
    assert affected["classification"] == {"evaluated": False, "success": None, "reason": "artifact_inspection_error"}
    assert all(not entry["artifact_errors"] for entry in index["entries"][1:])
    assert all(entry["classification"]["evaluated"] for entry in index["entries"][1:])
    expected_attempts = 1 if interrupt else 32
    groups = result.report["groups"].values()
    assert sum(group["total"] for group in groups) == expected_attempts
    assert sum(group["evaluated"] for group in groups) == expected_attempts - 1
    assert sum(group["successful"] for group in groups) == expected_attempts - 1
    assert sum(group["unevaluated"] for group in groups) == 1


@pytest.mark.parametrize(("unavailable", "interrupt"), [
    (("result.txt",), False), (("events.jsonl",), False), (("metadata.json",), False),
    (("result.txt", "events.jsonl", "metadata.json"), False), (("result.txt",), True),
])
def test_artifact_inspection_error_preserves_attempt_and_continues_real_fitd(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unavailable: tuple[str, ...], interrupt: bool,
) -> None:
    lib, kwargs, _ = local
    config = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    users: list[FITDExperiment] = []
    failed_stats = _install_inspection_fault(monkeypatch, users, unavailable, interrupt)
    previous_handler = signal.getsignal(signal.SIGINT)
    result = run_alert_series(MANIFEST, config, tmp_path / "runs", taxonomy_path=TAXONOMY, jobs=1)
    assert signal.getsignal(signal.SIGINT) is previous_handler
    assert failed_stats == list(unavailable)
    assert result.exit_code == (130 if interrupt else 1)
    assert result.report["complete"] is (not interrupt)
    _assert_inspected_runs(result, users, interrupt)
    _assert_inspection_errors(result, unavailable, interrupt)
    calls = len(lib.records.requests)
    assert reconstruct_report(result.directory / "index.json") == result.report
    assert json.loads((result.directory / "report.json").read_text()) == result.report
    assert len(lib.records.requests) == calls
