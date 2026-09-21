"""Artifact and held-subprocess assertions for installed CLI scenarios."""

import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from app_test_support import HttpPeer
from http_peer import Reply

from alert.values import object_value


def read_artifacts(root: Path):
    (directory,) = root.iterdir()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["kind"] == "run_finished"
    assert events[-1]["data"]["state"] == metadata["state"]
    assert events[-1]["data"]["reason"] == metadata["reason"]
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert all(event["run_id"] == directory.name for event in events)
    assert metadata["run_id"] == directory.name
    return directory, metadata, events


def assert_roles(endpoint: HttpPeer, settings: object) -> None:
    profile = object_value(settings, "settings")
    for role, temperature, timeout, auth in (
        ("Target", 0.25, 120, "Bearer synthetic-target-secret"),
        ("Attacker", 0.75, 15, "Custom synthetic-attacker-secret"),
    ):
        calls = [request for request in endpoint.requests if f"/{role.lower()}/" in request[0]]
        assert len(calls) >= 5
        assert all(request[1]["Authorization"] == auth for request in calls)
        assert all(request[2]["model"] == "puzzle-model" for request in calls)
        assert all(request[2]["temperature"] == temperature for request in calls)
        assert all(request[2]["n"] == 1 and request[2]["stream"] is False for request in calls)
        instructions = f"{role} puzzle " + ("keeper" if role == "Target" else "author")
        assert all(request[2]["messages"][0]["content"] == instructions for request in calls)
        assert object_value(profile[role], role)["timeout"] == timeout


@dataclass
class InterruptedCLI:
    stdout: str
    stderr: str
    prefix: bytes
    completed_ids: set[str]
    calls: int


def interrupt_cli(
    process: subprocess.Popen[str], endpoint: HttpPeer, held: Reply, root: Path, *, process_group: bool,
) -> InterruptedCLI:
    assert held.entered.wait(5), "CLI did not reach the held HTTP exchange"
    assert process.poll() is None
    calls = len(endpoint.requests)
    (directory,) = root.iterdir()
    prefix = (directory / "events.jsonl").read_bytes()
    finished = [json.loads(line) for line in prefix.splitlines()]
    completed_ids: set[str] = {event["data"]["call_id"] for event in finished if event["kind"] == "call_completed"}
    assert len(completed_ids) == calls - 1
    assert not (directory / "metadata.json").exists()
    assert not (directory / "result.txt").exists()
    if process_group:
        # Like terminal Ctrl+C: the foreground process group receives SIGINT.
        os.killpg(process.pid, signal.SIGINT)
    else:
        process.send_signal(signal.SIGINT)
    with pytest.raises(subprocess.TimeoutExpired):
        process.communicate(timeout=0.3)
    process.send_signal(signal.SIGINT)
    with pytest.raises(subprocess.TimeoutExpired):
        process.communicate(timeout=0.15)
    assert not held.disconnected.is_set()
    held.release.set()
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 130, stdout + stderr
    assert len(endpoint.requests) == calls
    return InterruptedCLI(stdout, stderr, prefix, completed_ids, calls)


def assert_stopped_cli(root: Path, evidence: InterruptedCLI, held: Reply, role: str, expected: str | None) -> None:
    directory, metadata, events = read_artifacts(root)
    assert metadata["state"] == "STOPPED"
    assert metadata["reason"] == "user_stop"
    assert "STOPPED" in evidence.stdout and "Traceback" not in evidence.stderr
    assert (directory / "events.jsonl").read_bytes().startswith(evidence.prefix)
    completions = [event["data"] for event in events if event["kind"] == "call_completed"]
    assert {event["call_id"] for event in completions[:-1]} == evidence.completed_ids
    assert completions[-1]["role"] == role
    assert completions[-1]["response"]["text"] == held.text
    assert not any(event["kind"] == "call_cancelled" for event in events)
    stopped = next(event for event in events if event["kind"] == "stop_requested")
    assert not any(event["kind"] == "call_started" and event["sequence"] > stopped["sequence"] for event in events)
    if expected is None:
        assert metadata["result"] is None and metadata["evaluation"] is None
        assert not (directory / "result.txt").exists()
        assert "отсутствует (None)" in evidence.stdout
    else:
        assert metadata["result"]["text"] == expected
        assert (directory / "result.txt").read_text() == expected
        assert metadata["evaluation"] == {"result_id": metadata["result"]["id"], "success": False}
        assert "Оценка результата: неуспех" in evidence.stdout
