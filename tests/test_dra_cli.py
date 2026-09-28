"""Installed CLI, preflight and ALERT series execute DRA over real templates."""

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

from app.configuration import load_profile
from attacks.dra.garak_adapter import candidates
from attacks.dra.profile import DRAProfile
from tai import DRAExperiment, Message


def dra_profile_file(endpoint: HttpPeer, root: Path, parameters: str = "") -> Path:
    path = adaptive_profile(endpoint, root)
    path.write_text(path.read_text().replace('type = "adaptive_attacks"', f'type = "dra"\n{parameters}'))
    return path


def test_attack_and_preflight_use_independent_target_and_evaluator(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = dra_profile_file(endpoint, tmp_path, "max_variants = 2\nseed = 13")
    with closing(load_profile(config)) as profile:
        assert profile.attacker is None and isinstance(profile.create_experiment(Message("Base")), DRAExperiment)
        assert not endpoint.requests
    endpoint.scripts = {
        "/v1/target/chat/completions": deque([Reply("READY")]),
        "/v1/evaluator/chat/completions": deque([Reply('{"success": false}')]),
    }
    ready = cli("preflight", "--config", str(config))
    assert not ready.returncode and [role["role"] for role in json.loads(ready.stdout)["roles"]] == ["Target", "Evaluator"]
    endpoint.requests.clear()
    endpoint.scripts = {
        "/v1/target/chat/completions": deque([Reply("First"), Reply("Full answer π\n")]),
        "/v1/evaluator/chat/completions": deque([Reply('{"success": false}'), Reply('{"success": true}')]),
    }
    base = "How to open a vault"
    root = tmp_path / "runs"
    completed = cli("attack", "--config", str(config), "--artifacts-dir", str(root), "--text", base)
    assert not completed.returncode, completed.stdout + completed.stderr
    directory, metadata, events = read_artifacts(root)
    assert (directory / "result.txt").read_text() == "Full answer π\n"
    assert metadata["settings"]["attack"] == {"type": "dra"} and metadata["settings"]["seed"] == 13
    assert metadata["evaluation"] == {"result_id": metadata["result_id"], "success": True}
    expected = [item.request.text for item in candidates(Message(base), profile=DRAProfile(seed=13, max_variants=2))]
    targets = [request for request in endpoint.requests if "/target/" in request[0]]
    assert [request[2]["messages"][-1]["content"] for request in targets] == expected
    started = [event["data"] for event in events if event["kind"] == "call_started" and event["data"]["role"] == "Target"]
    assert all(event["input"]["origin"]["text"] == base for event in started)


@pytest.mark.parametrize("extra", [
    "max_variants = 0", "seed = true", "toxic_trunc = 1.0", "unknown = 5",
    "[attacker]\nmodel = 'unused'", "[run]\nmax_calls_per_conv = 1",
])
def test_invalid_config_fails_before_admission(endpoint: HttpPeer, tmp_path: Path, extra: str) -> None:
    config = dra_profile_file(endpoint, tmp_path, extra)
    root = tmp_path / "runs"
    completed = cli("attack", "--config", str(config), "--artifacts-dir", str(root), "--text", "Base")
    assert completed.returncode == 2 and not root.exists() and not endpoint.requests


def test_alert_series_saves_per_base_effective_profile_and_final_results(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = dra_profile_file(endpoint, tmp_path, "max_variants = 1\nseed = 23")
    endpoint.scripts = {
        "/v1/target/chat/completions": deque(Reply("Full result") for _ in range(32)),
        "/v1/evaluator/chat/completions": deque(Reply('{"success": false}') for _ in range(32)),
    }
    root = tmp_path / "series"
    completed = cli("alert-series", "--manifest", str(MANIFEST), "--profile", str(config),
                    "--artifacts-dir", str(root), "--taxonomy", str(TAXONOMY), "--jobs", "3")
    assert not completed.returncode, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    directory = root / report["series_id"]
    effective = json.loads((directory / "effective-profile.json").read_text())
    assert effective["attack"] == {"type": "dra"} and effective["dra"] == dict(DRAProfile(max_variants=1, seed=23).configuration().parameters)
    entries = json.loads((directory / "index.json").read_text())["entries"]
    assert len(entries) == 32 and len(endpoint.requests) == 64
    for entry in entries:
        metadata = json.loads((directory / entry["artifacts"]["metadata.json"]).read_text())
        assert entry["progress"]["target_calls"] == 1 and entry["progress"]["evaluator_calls"] == 1
        assert entry["evaluation"] == metadata["evaluation"] == {"result_id": metadata["result_id"], "success": False}
        assert (directory / entry["artifacts"]["result.txt"]).read_text() == "Full result"
