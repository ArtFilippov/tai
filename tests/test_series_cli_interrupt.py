"""Installed series CLI waits for admitted exchanges and reconstructs offline."""

import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import pytest
from app_cli_checks import InterruptedCLI, interrupt_cli
from app_test_support import HttpPeer, cleanup_process
from http_peer import Reply
from series_support import MANIFEST, TAXONOMY
from test_model_profile import http_profile


def _assert_series_result(directory: Path, evidence: InterruptedCLI, role: str, state: str, success: bool | None) -> None:
    assert evidence.stderr.count("Backend: OpenAI-compatible") == 1
    assert f"#01 hate_body {state}" in evidence.stderr and "УСПЕХ" not in evidence.stderr
    report = json.loads(evidence.stdout)
    index = json.loads((directory / "index.json").read_text())
    assert len(index["entries"]) == 1 and index["interrupted"] is True
    entry = index["entries"][0]
    assert entry["state"] == state
    if success is not None:
        assert entry["evaluation"] == {"result_id": entry["result_id"], "success": success}
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
    events = [json.loads(line) for line in (directory / entry["artifacts"]["events.jsonl"]).read_text().splitlines()]
    assert len([event for event in events if event["kind"] == "call_completed"]) == evidence.calls
    assert not any(event["kind"].endswith("_cancelled") for event in events)
    assert report["groups"]["hate"] == {"total": 1, "evaluated": 0, "successful": 0,
                                         "unevaluated": 1, "success_percent": None}
    assert all("synthetic-" not in path.read_text() for path in directory.rglob("*.json"))


def _assert_rebuilt(directory: Path, executable: str, stdout: str) -> None:
    rebuilt = directory / "rebuilt.json"
    offline = subprocess.run([executable, "alert-report", "--index", str(directory / "index.json"),
                              "--output", str(rebuilt)], capture_output=True, text=True, timeout=10, check=False)
    assert not offline.returncode, offline.stderr
    assert json.loads(offline.stdout) == json.loads(stdout)
    assert rebuilt.read_bytes() == (directory / "report.json").read_bytes()


@pytest.mark.parametrize(("role", "response", "state", "success"), [
    ("target", "Returned", "STOPPED", None), ("evaluator", '{"success": true}', "STOPPED", True),
    ("evaluator", '{"success": false}', "STOPPED", False), ("evaluator", '{"success": "false"}', "FAILED", None),
])
def test_installed_cli_sigint_finishes_current_call_and_rebuilds_offline(
    endpoint: HttpPeer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    role: str, response: str, state: str, success: bool | None,
) -> None:
    profile = http_profile(endpoint, tmp_path, monkeypatch)
    profile.write_text(profile.read_text() + "\n[run]\nmax_calls_per_conv=20\n")
    held = Reply(response, hold=True)
    if role == "target":
        endpoint.scripts["/v1/target/chat/completions"][0] = held
    else:
        endpoint.scripts["/v1/evaluator/chat/completions"] = deque([held])
    command = [str(Path(sys.executable).with_name("tai")), "alert-series", "--jobs", "1", "--manifest", str(MANIFEST.resolve()),
               "--profile", str(profile), "--artifacts-dir", str(tmp_path / "runs"), "--taxonomy", str(TAXONOMY.resolve())]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
        try:
            evidence = _interrupt_series(process, endpoint, held, tmp_path / "runs")
        finally:
            cleanup_process(process, held)
    (directory,) = (tmp_path / "runs").iterdir()
    _assert_series_result(directory, evidence, role, state, success)
    _assert_rebuilt(directory, command[0], evidence.stdout)
    assert len(endpoint.requests) == evidence.calls


def _interrupt_series(process: subprocess.Popen[str], endpoint: HttpPeer, held: Reply, root: Path) -> InterruptedCLI:
    assert held.entered.wait(10), "series did not reach held exchange"
    # The series has a unique directory above its run directory.
    (directory,) = root.iterdir()
    return interrupt_cli(process, endpoint, held, directory / "runs", process_group=True)
