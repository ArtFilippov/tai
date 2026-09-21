"""SIGINT admission/finalization/persistence races and real CLI stop requests."""

import json
import os
import signal
import subprocess
import sys
from collections import deque
from pathlib import Path
from threading import Event, Thread

import pytest
from http_peer import Reply
from test_model_profile import http_profile
from test_qwen import local as local
from test_series import MANIFEST, TAXONOMY, positive_responses, qwen_profile

from tai import FITDUser
from tai._config import RoleProfile
from tai.fitd import _Execution
from tai.series import run_alert_series
from tai.series_report import reconstruct_report


@pytest.mark.parametrize(
    ("boundary", "attempts"),
    [
        ("construction", 0),
        ("initial_save", 0),
        ("before_start", 1),
        ("after_start", 1),
        ("finalization", 1),
        ("first_save", 1),
        ("last_save", 32),
    ],
)
def test_interrupt_races_preserve_actual_outcomes_and_attempted_prefix(
    local, tmp_path, monkeypatch, boundary, attempts
):
    import tai.series as series

    lib, kwargs, _ = local
    profile = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    raised = Event()
    factory, start, finalize = (
        RoleProfile.create_experiment,
        FITDUser.start,
        _Execution._finalize,
    )
    write = series._write_json

    def interrupt():
        if not raised.is_set():
            raised.set()
            # A real process signal, also when called from the FITD worker.
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGINT)

    def create(*args, **kwargs):
        user = factory(*args, **kwargs)
        if boundary == "construction":
            interrupt()
        return user

    def begin(user):
        if boundary == "before_start":
            interrupt()
        start(user)
        if boundary == "after_start":
            interrupt()

    def finish(execution, *args):
        if boundary == "finalization":
            interrupt()
            assert execution._stop_requested.wait(5)
        finalize(execution, *args)

    def persist(path, value):
        write(path, value)
        if path.name == "index.json":
            count = len(value["entries"])
            if (
                (boundary == "initial_save" and count == 0)
                or (boundary == "first_save" and count == 1)
                or (boundary == "last_save" and count == 32 and value["finished_at"])
            ):
                interrupt()

    monkeypatch.setattr(RoleProfile, "create_experiment", create)
    monkeypatch.setattr(FITDUser, "start", begin)
    monkeypatch.setattr(_Execution, "_finalize", finish)
    monkeypatch.setattr(series, "_write_json", persist)
    previous = signal.getsignal(signal.SIGINT)
    result = run_alert_series(
        MANIFEST, profile, tmp_path / "runs", taxonomy_path=TAXONOMY
    )
    assert raised.is_set()
    assert signal.getsignal(signal.SIGINT) is previous
    assert result.exit_code == 130 and result.report["complete"] is False
    index = json.loads((result.directory / "index.json").read_text())
    assert index["interrupted"] is True and index["complete"] is False
    assert index["finished_at"] is not None and len(index["entries"]) == attempts
    assert sum(g["total"] for g in result.report["groups"].values()) == attempts
    assert reconstruct_report(result.directory / "index.json") == result.report
    assert json.loads((result.directory / "report.json").read_text()) == result.report
    for entry in index["entries"]:
        metadata = json.loads(
            (result.directory / entry["artifacts"]["metadata.json"]).read_text()
        )
        assert entry["state"] == metadata["state"]
        assert entry["reason"] == metadata["reason"]
        assert entry["evaluation"] == metadata["evaluation"]
    if boundary == "finalization":
        entry = index["entries"][0]
        assert entry["state"] == "STOPPED" and entry["reason"] == "user_stop"
        assert entry["evaluation"]["success"] is True
        assert entry["classification"]["evaluated"] is False
        assert result.report["groups"]["hate"]["success_percent"] is None
    if boundary in ("first_save", "last_save"):
        assert all(e["state"] == "COMPLETED" for e in index["entries"])


@pytest.mark.parametrize("phase", ["construction", "after_start"])
def test_creation_or_post_admission_error_is_saved_and_next_run_waits(
    local, tmp_path, monkeypatch, phase
):
    lib, kwargs, _ = local
    profile = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    factory, start = RoleProfile.create_experiment, FITDUser.start
    users = []
    constructed = 0

    def create(*args):
        nonlocal constructed
        constructed += 1
        if users and users[-1].status().run_id is not None:
            assert users[-1].wait(0)
        if phase == "construction" and constructed == 2:
            raise RuntimeError("factory unavailable")
        user = factory(*args)
        users.append(user)
        return user

    def begin(user):
        start(user)
        if phase == "after_start" and constructed == 2:
            raise RuntimeError("start admitted before raising")

    monkeypatch.setattr(RoleProfile, "create_experiment", create)
    monkeypatch.setattr(FITDUser, "start", begin)
    result = run_alert_series(
        MANIFEST, profile, tmp_path / "runs", taxonomy_path=TAXONOMY
    )
    index = json.loads((result.directory / "index.json").read_text())
    assert result.exit_code == 1 and result.report["complete"] is True
    assert len(index["entries"]) == constructed == 32
    failed = index["entries"][1]
    assert failed["error"]["phase"] == (
        "create_experiment" if phase == "construction" else "start"
    )
    assert failed["classification"]["evaluated"] is False
    if phase == "construction":
        assert failed["run_id"] is None and failed["state"] is None
    else:
        assert failed["run_id"] == users[1].status().run_id
        assert users[1].wait(0)
    assert index["entries"][2]["state"] == "COMPLETED"


def test_terminal_status_before_worker_cleanup_does_not_admit_next_run(
    local, tmp_path, monkeypatch
):
    import tai.fitd as fitd

    lib, kwargs, _ = local
    profile = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    entered, release, checked = Event(), Event(), Event()
    release_chats = fitd._release_chats
    users = []
    factory = RoleProfile.create_experiment

    def create(*args):
        user = factory(*args)
        if users:
            assert checked.is_set() and users[-1].wait(0)
        users.append(user)
        return user

    def cleanup(chats, owner):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        release_chats(chats, owner)

    def control():
        try:
            assert entered.wait(5)
            assert len(users) == 1
            assert users[0].status().state == "COMPLETED"
            assert not users[0].wait(0)
            checked.set()
        finally:
            release.set()

    monkeypatch.setattr(RoleProfile, "create_experiment", create)
    monkeypatch.setattr(fitd, "_release_chats", cleanup)
    controller = Thread(target=control)
    controller.start()
    try:
        result = run_alert_series(
            MANIFEST, profile, tmp_path / "runs", taxonomy_path=TAXONOMY
        )
    finally:
        release.set()
        controller.join(10)
    assert checked.is_set() and result.exit_code == 0
    assert len(users) == 32


@pytest.mark.parametrize(
    ("role", "response", "state", "success"),
    [
        ("target", "Returned", "STOPPED", None),
        ("evaluator", '{"success": true}', "STOPPED", True),
        ("evaluator", '{"success": false}', "STOPPED", False),
        ("evaluator", '{"success": "false"}', "FAILED", None),
    ],
)
def test_installed_cli_sigint_finishes_current_call_and_rebuilds_offline(
    endpoint, tmp_path, monkeypatch, role, response, state, success
):
    profile = http_profile(endpoint, tmp_path, monkeypatch)
    profile.write_text(profile.read_text() + "\n[run]\nmax_calls_per_conv=20\n")
    held = Reply(response, hold=True)
    if role == "target":
        endpoint.scripts["/v1/target/chat/completions"][0] = held
    else:
        endpoint.scripts["/v1/evaluator/chat/completions"] = deque([held])
    command = [
        str(Path(sys.executable).with_name("tai")),
        "alert-series",
        "--manifest",
        str(MANIFEST.resolve()),
        "--profile",
        str(profile),
        "--artifacts-dir",
        str(tmp_path / "runs"),
        "--taxonomy",
        str(TAXONOMY.resolve()),
    ]
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            assert held.entered.wait(10), "series did not reach held exchange"
            calls = len(endpoint.requests)
            os.killpg(process.pid, signal.SIGINT)
            with pytest.raises(subprocess.TimeoutExpired):
                process.communicate(timeout=0.3)
            os.killpg(process.pid, signal.SIGINT)
            with pytest.raises(subprocess.TimeoutExpired):
                process.communicate(timeout=0.15)
            assert not held.disconnected.is_set()
            held.release.set()
            stdout, stderr = process.communicate(timeout=5)
            assert process.returncode == 130, stdout + stderr
            assert len(endpoint.requests) == calls
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            held.release.set()
    (directory,) = (tmp_path / "runs").iterdir()
    report = json.loads(stdout)
    index = json.loads((directory / "index.json").read_text())
    assert len(index["entries"]) == 1 and index["interrupted"] is True
    entry = index["entries"][0]
    assert entry["state"] == state
    if success is not None:
        assert entry["evaluation"] == {
            "result_id": entry["result_id"],
            "success": success,
        }
    else:
        assert entry["evaluation"] is None
    if role == "target":
        assert entry["result"] is None
    else:
        assert entry["result"]["id"] == entry["result_id"]
    metadata = json.loads((directory / entry["artifacts"]["metadata.json"]).read_text())
    assert metadata["state"] == entry["state"]
    assert metadata["reason"] == entry["reason"]
    assert metadata["evaluation"] == entry["evaluation"]
    events = [
        json.loads(line)
        for line in (directory / entry["artifacts"]["events.jsonl"])
        .read_text()
        .splitlines()
    ]
    assert len([e for e in events if e["kind"] == "call_completed"]) == calls
    assert not any(e["kind"].endswith("_cancelled") for e in events)
    assert report["groups"]["hate"] == {
        "total": 1,
        "evaluated": 0,
        "successful": 0,
        "unevaluated": 1,
        "success_percent": None,
    }
    assert all("synthetic-" not in p.read_text() for p in directory.rglob("*.json"))
    rebuilt = directory / "rebuilt.json"
    offline = subprocess.run(
        [
            command[0],
            "alert-report",
            "--index",
            str(directory / "index.json"),
            "--output",
            str(rebuilt),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert offline.returncode == 0, offline.stderr
    assert json.loads(offline.stdout) == report
    assert rebuilt.read_bytes() == (directory / "report.json").read_bytes()
    assert len(endpoint.requests) == calls
