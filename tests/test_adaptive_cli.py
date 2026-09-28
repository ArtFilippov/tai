"""Installed subprocess command selection, input validation and role readiness."""

import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import pytest
from adaptive_app_support import adaptive_profile, cli
from app_cli_checks import assert_stopped_cli, interrupt_cli, read_artifacts
from app_test_support import HttpPeer, cleanup_process, session
from http_peer import Reply

from app.configuration import load_profile
from attacks.adaptive_attacks.garak_adapter import variants
from tai import AdaptiveAttacksExperiment, FITDExperiment, Message


@pytest.mark.parametrize("source", ["text", "text-file"])
def test_installed_attack_command_runs_real_variants_and_saves_second_answer(endpoint: HttpPeer, tmp_path: Path, source: str) -> None:
    config = adaptive_profile(endpoint, tmp_path)
    endpoint.scripts = {
        "/v1/target/chat/completions": deque([Reply("First"), Reply("Full second answer π\n")]),
        "/v1/evaluator/chat/completions": deque([Reply('{"success": false}'), Reply('{"success": true}')]),
    }
    base = "Explain {goal}\r\nAnd {suffix} π\n"
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
    assert metadata["settings"]["attack"] == {"type": "adaptive_attacks"}
    assert metadata["evaluation"] == {"result_id": metadata["result_id"], "success": True}
    assert (directory / "result.txt").read_text() == "Full second answer π\n"
    targets = [request for request in endpoint.requests if "/target/" in request[0]]
    assert len(targets) == 2 and len(endpoint.requests) == 4
    assert len({session(request) for request in endpoint.requests}) == 4
    assert [request[2]["messages"][-1]["content"] for request in targets] == [message.text for message in variants(Message(base))[:2]]
    assert all(len(request[2]["messages"]) == 2 for request in targets)
    assert all(request[2]["messages"][0]["content"] == "Target system" for request in targets)
    assert "Target=2" in completed.stdout and "Full second answer" in completed.stdout
    assert "до первого успеха" in completed.stdout
    assert "Лимит FITD" not in completed.stdout and "Attacker=" not in completed.stdout
    assert sum(event["kind"] == "evaluation_completed" for event in events) == 2


def test_profile_reading_is_lazy_and_preflight_checks_only_selected_roles(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = adaptive_profile(endpoint, tmp_path)
    profile = load_profile(config)
    assert profile.attacker is None
    assert isinstance(profile.create_experiment(Message("Base")), AdaptiveAttacksExperiment)
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


@pytest.mark.parametrize("extra", [
    '[attack]\ntype="unknown"', '[attack]\ntype="adaptive_attacks"\nunknown=1',
    '[attack]\ntype="adaptive_attacks"\n[run]\nmax_calls_per_conv=5',
    '[attack]\ntype="adaptive_attacks"\n[run]\nvariants=2',
    '[attack]\ntype="adaptive_attacks"\n[run]\ngenerations=2',
    '[attack]\ntype="adaptive_attacks"\n[attacker]\nmodel="unused"',
    '[attack]',
])
def test_invalid_attack_profiles_fail_before_admission(endpoint: HttpPeer, tmp_path: Path, extra: str) -> None:
    config = adaptive_profile(endpoint, tmp_path)
    config.write_text(config.read_text().replace('[attack]\ntype = "adaptive_attacks"', extra))
    root = tmp_path / "runs"
    completed = cli("attack", "--config", str(config), "--text", "Base", "--artifacts-dir", str(root))
    assert completed.returncode == 2 and "Traceback" not in completed.stderr
    assert not endpoint.requests and not root.exists()


def test_fitd_command_rejects_an_adaptive_profile(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = adaptive_profile(endpoint, tmp_path)
    root = tmp_path / "runs"
    completed = cli("fitd", "--config", str(config), "--text", "Base", "--artifacts-dir", str(root))
    assert completed.returncode == 2 and "tai attack" in completed.stderr
    assert not endpoint.requests and not root.exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_attack_command_preserves_legacy_and_explicit_fitd_profiles(endpoint: HttpPeer, tmp_path: Path, explicit: bool) -> None:
    config = adaptive_profile(endpoint, tmp_path)
    config.write_text(config.read_text().replace('[attack]\ntype = "adaptive_attacks"', '[attack]\ntype = "fitd"' if explicit else "") + f'''
[attacker]
base_url = "{endpoint.url}/attacker"
model = "controlled"
[run]
max_calls_per_conv = 1
''')
    profile = load_profile(config)
    assert isinstance(profile.create_experiment(Message("Base")), FITDExperiment)
    endpoint.scripts = {
        "/v1/target/chat/completions": deque([Reply("First")]),
        "/v1/attacker/chat/completions": deque(Reply(text) for text in ("Initial", "L1", "L2", "L3", "yes")),
    }
    root = tmp_path / "runs"
    completed = cli("attack", "--config", str(config), "--text", "Base", "--artifacts-dir", str(root))
    assert not completed.returncode, completed.stdout + completed.stderr
    _, metadata, _ = read_artifacts(root)
    assert metadata["settings"]["attack"] == {"type": "fitd"}
    assert metadata["reason"] == "round_limit" and metadata["progress"]["target_calls"] == 1
    assert "Лимит FITD: 1" in completed.stdout and "Attacker=" in completed.stdout


@pytest.mark.parametrize("role", ["Target", "Evaluator"])
def test_attack_subprocess_sigint_uses_existing_foreground_control(endpoint: HttpPeer, tmp_path: Path, role: str) -> None:
    config = adaptive_profile(endpoint, tmp_path)
    held = Reply('{"success": false}' if role == "Evaluator" else "Unaccepted", hold=True)
    endpoint.scripts = {
        "/v1/target/chat/completions": deque([held if role == "Target" else Reply("Presented")]),
        "/v1/evaluator/chat/completions": deque([held]),
    }
    root = tmp_path / "runs"
    with subprocess.Popen(
        [str(Path(sys.executable).with_name("tai")), "attack", "--config", str(config), "--text", "Base", "--artifacts-dir", str(root)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", start_new_session=True,
    ) as process:
        try:
            evidence = interrupt_cli(process, endpoint, held, root, process_group=True)
        finally:
            cleanup_process(process, held)
    assert_stopped_cli(root, evidence, held, role, "Presented" if role == "Evaluator" else None)
