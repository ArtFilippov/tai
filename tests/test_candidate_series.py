"""Real garak, installed ALERT CLI, concurrent local failures and multi-User SIGINT."""

import json
import signal
import subprocess
import sys
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from adaptive_app_support import adaptive_profile, cli
from app_test_support import HttpPeer, cleanup_process, session
from badchars_app_support import badchars_profile
from http_peer import Reply
from series_support import MANIFEST, TAXONOMY

from alert.values import object_value
from attacks.adaptive_attacks.garak_adapter import variants
from attacks.badchars.garak_adapter import candidates
from attacks.badchars.profile import DeletionProfile
from tai import Message


@contextmanager
def _series(config: Path, root: Path, held: list[Reply]) -> Generator[subprocess.Popen[str]]:
    command = [str(Path(sys.executable).with_name("tai")), "alert-series", "--jobs", "3",
               "--manifest", str(MANIFEST), "--profile", str(config),
               "--taxonomy", str(TAXONOMY), "--artifacts-dir", str(root)]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", start_new_session=True) as process:
        try:
            yield process
        finally:
            for reply in held:
                reply.release.set()
            cleanup_process(process, held[0])


def _check_saved(root: Path, stdout: str, endpoint: HttpPeer, strategy: str) -> list[dict[str, object]]:
    report = json.loads(stdout)
    directory = root / report["series_id"]
    index = json.loads((directory / "index.json").read_text())
    effective = json.loads((directory / "effective-profile.json").read_text())
    assert effective["attack"] == {"type": strategy} and "Attacker" not in effective
    for entry in index["entries"]:
        _check_entry(directory, entry, strategy)
    requests = len(endpoint.requests)
    rebuilt = directory / "rebuilt.json"
    result = cli("alert-report", "--index", str(directory / "index.json"), "--output", str(rebuilt))
    assert not result.returncode and json.loads(result.stdout) == report
    assert rebuilt.read_bytes() == (directory / "report.json").read_bytes()
    assert len(endpoint.requests) == requests
    assert len({session(request) for request in endpoint.requests}) == requests
    assert all(len(request[2]["messages"]) == 2 for request in endpoint.requests)
    return index["entries"]


def _check_entry(directory: Path, value: object, strategy: str) -> None:
    entry = object_value(value, "entry")
    artifacts = object_value(entry["artifacts"], "artifacts")
    metadata = json.loads((directory / str(artifacts["metadata.json"])).read_text())
    assert metadata["state"] == entry["state"] and metadata["evaluation"] == entry["evaluation"]
    assert metadata["settings"]["attack"] == {"type": strategy}
    assert "Attacker" not in metadata["settings"]
    if metadata["result"] is not None:
        assert metadata["result_id"] == metadata["result"]["id"]
        assert (directory / str(artifacts["result.txt"])).read_text() == metadata["result"]["text"]
    events = [json.loads(line) for line in (directory / str(artifacts["events.jsonl"])).read_text().splitlines()]
    started = [event for event in events if event["kind"] == "call_started"]
    target = next(event["data"] for event in started if event["data"]["role"] == "Target")
    base = Message(metadata["settings"]["base"]["text"])
    expected = variants(base)[0] if strategy == "adaptive_attacks" else next(candidates(base, profile=DeletionProfile())).request
    assert target["input"]["text"] == expected.text
    assert target["input"]["origin"]["id"] == entry["base_id"]
    assert any(event["kind"] == "resources_released" for event in events)
    assert events[-1]["kind"] == "run_finished"
    boundaries = [event["sequence"] for event in events if event["kind"] == "stop_requested" or (
        event["kind"] == "evaluation_completed" and event["data"]["success"] is True)]
    if boundaries:
        assert all(event["sequence"] < min(boundaries) for event in started)


@pytest.mark.parametrize("strategy", ["adaptive_attacks", "badchars_deletion"])
def test_parallel_series_saves_later_run_and_continues_after_local_error(
    endpoint: HttpPeer, tmp_path: Path, strategy: str,
) -> None:
    config = (adaptive_profile if strategy == "adaptive_attacks" else badchars_profile)(endpoint, tmp_path)
    held = [Reply("Accepted", hold=True), Reply("Local failure", status=500, hold=True), Reply("Accepted", hold=True)]
    following = Reply("Accepted")
    endpoint.scripts = {
        "/v1/target/chat/completions": deque([*held, following, *(Reply("Accepted") for _ in range(28))]),
        "/v1/evaluator/chat/completions": deque(Reply('{"success": true}') for _ in range(31)),
    }
    root = tmp_path / "series"
    with _series(config, root, held) as process:
        for reply in held:
            assert reply.entered.wait(10)
        assert len(endpoint.requests) == 3  # Three real calls overlap; the fourth cannot start yet.
        held[2].release.set()
        assert following.entered.wait(10)  # Saving releases the later run's slot first.
        assert not held[0].disconnected.is_set() and not held[1].disconnected.is_set()
        for reply in held:
            reply.release.set()
        stdout, stderr = process.communicate(timeout=45)
        assert process.returncode == 1, stdout + stderr
    assert "Backend: OpenAI-compatible" in stderr and "FAILED" in stderr and "УСПЕХ" in stderr
    entries = _check_saved(root, stdout, endpoint, strategy)
    assert len(entries) == 32 and sum(entry["state"] == "FAILED" for entry in entries) == 1
    assert sum(entry["state"] == "COMPLETED" for entry in entries) == 31
    report = json.loads(stdout)
    assert report["complete"] is True
    assert sum(group["total"] for group in report["groups"].values()) == 32
    assert sum(group["evaluated"] for group in report["groups"].values()) == 31
    assert sum(group["successful"] for group in report["groups"].values()) == 31
    assert len(endpoint.requests) == 63  # No variant after the first successful evaluation.


@pytest.mark.parametrize("strategy", ["adaptive_attacks", "badchars_deletion"])
@pytest.mark.parametrize("role", ["target", "evaluator"])
def test_parallel_sigint_drains_all_admitted_calls_and_preserves_final_evaluations(
    endpoint: HttpPeer, tmp_path: Path, strategy: str, role: str,
) -> None:
    config = (adaptive_profile if strategy == "adaptive_attacks" else badchars_profile)(endpoint, tmp_path)
    held = [Reply('{"success": true}' if role == "evaluator" else "Unaccepted", hold=True) for _ in range(3)]
    endpoint.scripts = {
        "/v1/target/chat/completions": deque(held if role == "target" else [Reply("Presented") for _ in held]),
        "/v1/evaluator/chat/completions": deque(held if role == "evaluator" else []),
    }
    root = tmp_path / "series"
    with _series(config, root, held) as process:
        for reply in held:
            assert reply.entered.wait(10)
        for _ in range(2):
            process.send_signal(signal.SIGINT)
            with pytest.raises(subprocess.TimeoutExpired):
                process.communicate(timeout=0.2)
        assert not any(reply.disconnected.is_set() for reply in held)
        for reply in held:
            reply.release.set()
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, stdout + stderr
    assert "STOPPED" in stderr and "УСПЕХ" not in stderr
    entries = _check_saved(root, stdout, endpoint, strategy)
    assert len(entries) == 3 and all(entry["state"] == "STOPPED" for entry in entries)
    for entry in entries:
        assert entry["evaluation"] == ({"result_id": entry["result_id"], "success": True} if role == "evaluator" else None)
        assert (entry["result"] is None) == (role == "target")
    report = json.loads(stdout)
    assert report["complete"] is False
    assert sum(group["total"] for group in report["groups"].values()) == 3
    assert not sum(group["evaluated"] for group in report["groups"].values())
    assert len(endpoint.requests) == (6 if role == "evaluator" else 3)
