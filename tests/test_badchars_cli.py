"""Installed CLI, parameter validation and Target-only readiness for deletion."""

import json
from collections import deque
from contextlib import closing
from pathlib import Path

import pytest
from adaptive_app_support import cli
from app_cli_checks import read_artifacts
from app_test_support import HttpPeer, session
from badchars_app_support import badchars_profile
from http_peer import Reply

from app.configuration import load_profile
from attacks.badchars.garak_adapter import candidates
from attacks.badchars.profile import DeletionProfile
from tai import BadCharactersDeletionExperiment, Message


@pytest.mark.parametrize("source", ["text", "text-file"])
def test_installed_attack_command_preserves_backspace_and_saves_second_evaluated_response(
    endpoint: HttpPeer, tmp_path: Path, source: str,
) -> None:
    config = badchars_profile(endpoint, tmp_path, "max_ascii_variants = 2\nmax_position_candidates = 3")
    endpoint.scripts = {
        "/v1/target/chat/completions": deque([Reply("First"), Reply("Full second answer π\n")]),
        "/v1/evaluator/chat/completions": deque([Reply('{"success": false}'), Reply('{"success": true}')]),
    }
    base = "Explain {goal}\r\nAnd π🙂\n"
    argument = base
    if source == "text-file":
        path = tmp_path / "base text.txt"
        path.write_bytes(base.encode("utf-8"))
        argument = str(path)
    root = tmp_path / "runs"
    completed = cli("attack", "--config", str(config), "--artifacts-dir", str(root), f"--{source}", argument)
    assert not completed.returncode, completed.stdout + completed.stderr
    directory, metadata, events = read_artifacts(root)
    assert metadata["reason"] == "evaluator_success"
    assert metadata["settings"]["base"]["text"] == base
    assert metadata["settings"]["attack"] == {"type": "badchars_deletion"}
    assert metadata["settings"]["max_ascii_variants"] == 2 and metadata["settings"]["max_position_candidates"] == 3
    assert metadata["evaluation"] == {"result_id": metadata["result_id"], "success": True}
    assert (directory / "result.txt").read_text() == "Full second answer π\n"
    _assert_requests(endpoint, base)
    started = [event["data"] for event in events if event["kind"] == "call_started" and event["data"]["role"] == "Target"]
    assert [event["input"]["text"] for event in started] == [" \b" + base, "~\b" + base]
    assert all(event["input"]["origin"]["text"] == base for event in started)
    assert started[1]["context"]["cause"]["metadata"]["operations"] == {"positions": [0], "ascii_codes": [126]}
    assert "Target=2" in completed.stdout and "Full second answer" in completed.stdout


def _assert_requests(endpoint: HttpPeer, base: str) -> None:
    targets = [request for request in endpoint.requests if "/target/" in request[0]]
    assert len(targets) == 2 and len(endpoint.requests) == 4
    assert len({session(request) for request in endpoint.requests}) == 4
    expected = list(candidates(Message(base), profile=DeletionProfile(max_ascii_variants=2, max_position_candidates=3)))
    assert [request[2]["messages"][-1]["content"] for request in targets] == [item.request.text for item in expected[:2]]
    assert all(len(request[2]["messages"]) == 2 for request in targets)
    assert all(request[2]["messages"][0]["content"] == "Target system" for request in targets)


def test_profile_is_lazy_and_preflight_has_no_attacker(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = badchars_profile(endpoint, tmp_path)
    with closing(load_profile(config)) as profile:
        assert profile.attacker is None
        assert isinstance(profile.create_experiment(Message("Base")), BadCharactersDeletionExperiment)
        assert not endpoint.requests
    endpoint.scripts = {
        "/v1/target/chat/completions": deque([Reply("READY")]),
        "/v1/evaluator/chat/completions": deque([Reply('{"success": false}')]),
    }
    completed = cli("preflight", "--config", str(config))
    assert not completed.returncode, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert [role["role"] for role in report["roles"]] == ["Target", "Evaluator"]
    assert report["ready"] is True and len(endpoint.requests) == 2


@pytest.mark.parametrize("name", ["perturbation_budget", "max_position_candidates", "max_ascii_variants"])
@pytest.mark.parametrize("value", ["true", "false", "0", "-1", "1.5", '"1"', "[]", "{}"])
def test_toml_rejects_invalid_parameter_types_and_values(endpoint: HttpPeer, tmp_path: Path, name: str, value: str) -> None:
    config = badchars_profile(endpoint, tmp_path, f"{name} = {value}")
    with pytest.raises(ValueError, match=name):
        load_profile(config)
    assert not endpoint.requests


@pytest.mark.parametrize("parameters", [
    "max_ascii_variants = 96", "unknown = 1", 'category = "invisible"',
    "[run]\nmax_calls_per_conv = 1", "[attacker]\nmodel = 'unused'",
])
def test_invalid_profiles_fail_before_admission(endpoint: HttpPeer, tmp_path: Path, parameters: str) -> None:
    config = badchars_profile(endpoint, tmp_path, parameters)
    root = tmp_path / "runs"
    completed = cli("attack", "--config", str(config), "--text", "Base", "--artifacts-dir", str(root))
    assert completed.returncode == 2 and "Traceback" not in completed.stderr
    assert not endpoint.requests and not root.exists()


def test_fitd_command_rejects_deletion_profile(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = badchars_profile(endpoint, tmp_path)
    root = tmp_path / "runs"
    completed = cli("fitd", "--config", str(config), "--text", "Base", "--artifacts-dir", str(root))
    assert completed.returncode == 2 and "tai attack" in completed.stderr
    assert not endpoint.requests and not root.exists()


def test_ready_qwen_example_uses_default_deletion_parameters_without_inference() -> None:
    with closing(load_profile(Path("examples/badchars-deletion-qwen.toml"))) as profile:
        assert profile.strategy().settings() == DeletionProfile().configuration().settings()
        assert profile.attacker is None and tuple(profile.chats()) == ("Target", "Evaluator")
