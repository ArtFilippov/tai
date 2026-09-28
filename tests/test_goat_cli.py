"""CLI and ALERT use the same GOAT participants, limits and artifacts."""

import json
from collections import deque
from contextlib import closing
from pathlib import Path

import pytest
from adaptive_app_support import adaptive_profile, cli
from app_cli_checks import read_artifacts
from app_test_support import HttpPeer
from http_peer import Reply
from series_support import MANIFEST, TAXONOMY
from test_goat import otsr

from app.configuration import load_profile
from tai import GOATExperiment, Message


def goat_config(endpoint: HttpPeer, root: Path, rounds: int = 1) -> Path:
    path = adaptive_profile(endpoint, root)
    contents = path.read_text().replace('type = "adaptive_attacks"', 'type = "goat"')
    contents += f'''\n[attacker]
base_url = "{endpoint.url}/attacker"
model = "controlled"
opencode_session = true
[run]
max_calls_per_conv = {rounds}
'''
    path.write_text(contents)
    return path


def scripts(endpoint: HttpPeer, count: int) -> None:
    endpoint.scripts = {
        "/v1/attacker/chat/completions": deque(
            reply for _ in range(count) for reply in (Reply(otsr("Try this")), Reply("safe"))
        ),
        "/v1/target/chat/completions": deque(Reply("Full result") for _ in range(count)),
        "/v1/evaluator/chat/completions": deque(Reply('{"success": false}') for _ in range(count)),
    }


def test_preflight_and_attack_save_goat_evidence(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = goat_config(endpoint, tmp_path)
    with closing(load_profile(config)) as profile:
        assert profile.attacker is not None and isinstance(profile.create_experiment(Message("Base")), GOATExperiment)
        assert not endpoint.requests
    endpoint.scripts = {role: deque([Reply("READY")]) for role in (
        "/v1/attacker/chat/completions", "/v1/target/chat/completions",
    )}
    endpoint.scripts["/v1/evaluator/chat/completions"] = deque([Reply('{"success": false}')])
    ready = cli("preflight", "--config", str(config))
    assert not ready.returncode, ready.stderr
    assert [entry["role"] for entry in json.loads(ready.stdout)["roles"]] == ["Target", "Attacker", "Evaluator"]
    endpoint.requests.clear()
    scripts(endpoint, 1)
    root = tmp_path / "runs"
    completed = cli("attack", "--config", str(config), "--artifacts-dir", str(root), "--text", "Base request")
    assert not completed.returncode, completed.stdout + completed.stderr
    directory, metadata, events = read_artifacts(root)
    assert (directory / "result.txt").read_text() == "Full result"
    assert metadata["settings"]["attack"] == {"type": "goat"}
    assert metadata["settings"]["max_calls_per_conv"] == 1
    assert any(event["kind"] == "local_decision" and event["data"]["decision"] == "goat_judge" for event in events)
    assert [request[2]["messages"][-1]["content"] for request in endpoint.requests if "/target/" in request[0]] == ["Try this"]


@pytest.mark.parametrize("invalid", ["max_calls_per_conv = 0", "max_calls_per_conv = true", "max_calls_per_conv = -1"])
def test_invalid_limit_rejected_before_artifacts(endpoint: HttpPeer, tmp_path: Path, invalid: str) -> None:
    config = goat_config(endpoint, tmp_path)
    config.write_text(config.read_text().replace("max_calls_per_conv = 1", invalid))
    root = tmp_path / "runs"
    completed = cli("attack", "--config", str(config), "--artifacts-dir", str(root), "--text", "Base")
    assert completed.returncode == 2 and not root.exists() and not endpoint.requests


def test_alert_series_uses_one_explicit_base_per_run(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = goat_config(endpoint, tmp_path)
    scripts(endpoint, 32)
    root = tmp_path / "series"
    completed = cli("alert-series", "--manifest", str(MANIFEST), "--profile", str(config),
                    "--artifacts-dir", str(root), "--taxonomy", str(TAXONOMY), "--jobs", "1")
    assert not completed.returncode, completed.stdout + completed.stderr
    directory = root / json.loads(completed.stdout)["series_id"]
    effective = json.loads((directory / "effective-profile.json").read_text())
    assert effective["attack"] == {"type": "goat"} and effective["goat"]["max_calls_per_conv"] == 1
    entries = json.loads((directory / "index.json").read_text())["entries"]
    assert len(entries) == 32 and len(endpoint.requests) == 128
    for entry in entries:
        metadata = json.loads((directory / entry["artifacts"]["metadata.json"]).read_text())
        assert metadata["settings"]["attack"] == {"type": "goat"}
        assert (directory / entry["artifacts"]["result.txt"]).read_text() == "Full result"
