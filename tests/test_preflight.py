"""Readiness uses the installed CLI and the same role profile as tai fitd."""

import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import pytest
from http_peer import Reply
from test_cli import configure, read_artifacts
from test_zen_sessions import session


def profile(endpoint, tmp_path, monkeypatch):
    monkeypatch.setenv("TAI_TEST_TARGET_KEY", "synthetic-target-secret")
    monkeypatch.setenv("TAI_TEST_ATTACKER_AUTH", "Custom synthetic-attacker-secret")
    config = configure(endpoint, tmp_path)
    config.write_text(
        config.read_text()
        .replace("[target]\n", "[target]\nopencode_session=true\n")
        .replace("[attacker]\n", "[attacker]\nopencode_session=true\n")
    )
    endpoint.require_session = True
    return config


def preflight(config, *extra):
    return subprocess.run(
        [
            str(Path(sys.executable).with_name("tai")),
            "preflight",
            "--config",
            str(config),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_preflight_checks_both_configured_roles_and_saves_safe_report(
    endpoint, tmp_path, monkeypatch
):
    config = profile(endpoint, tmp_path, monkeypatch)
    endpoint.scripts = {
        "/v1/target/chat/completions": deque([Reply("READY")]),
        "/v1/attacker/chat/completions": deque([Reply("READY")]),
    }
    report = tmp_path / "readiness.json"
    completed = preflight(config, "--report", str(report))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    data = json.loads(completed.stdout)
    assert json.loads(report.read_text()) == data
    assert data["ready"] is True
    assert [r["role"] for r in data["roles"]] == ["Target", "Attacker", "Evaluator"]
    assert [r["response"] for r in data["roles"]] == ["READY", "READY", None]
    assert data["roles"][2]["ok"] is True
    assert data["roles"][2]["success"] is False
    assert len(endpoint.requests) == 2
    assert len({session(r) for r in endpoint.requests}) == 2
    for result, request, temperature, auth in zip(
        data["roles"][:2],
        endpoint.requests,
        [0.25, 0.75],
        ["Bearer synthetic-target-secret", "Custom synthetic-attacker-secret"],
        strict=True,
    ):
        assert result["conversation_id"] == session(request)
        assert result["settings"]["generation"]["temperature"] == temperature
        assert request[1]["Authorization"] == auth
        assert request[2]["temperature"] == temperature
        assert request[2]["messages"][-1] == {
            "role": "user",
            "content": "Reply with exactly READY.",
        }
        assert len(request[2]["messages"]) == 2
    assert "synthetic-" not in completed.stdout + completed.stderr
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("failing_role", ["target", "attacker"])
@pytest.mark.parametrize(
    "failure", ["http", "empty", "whitespace", "models", "truncated"]
)
def test_preflight_reports_each_role_even_when_one_cannot_generate(
    endpoint, tmp_path, monkeypatch, failing_role, failure
):
    config = profile(endpoint, tmp_path, monkeypatch)
    endpoint.scripts = {
        f"/v1/{role}/chat/completions": deque(
            [Reply("READY synthetic-target-secret synthetic-attacker-secret")]
        )
        for role in ("target", "attacker")
    }
    endpoint.scripts[f"/v1/{failing_role}/chat/completions"] = deque(
        [
            {
                "http": Reply(status=401, payload=b"synthetic-target-secret"),
                "empty": Reply(""),
                "whitespace": Reply(" \n\t"),
                "models": Reply(payload=b'{"data": [{"id": "big-pickle"}]}'),
                "truncated": Reply(
                    payload=b"synthetic-attacker-secret", truncated=True
                ),
            }[failure]
        ]
    )
    completed = preflight(config)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    data = json.loads(completed.stdout)
    assert data["ready"] is False
    assert len(endpoint.requests) == 2
    for result in data["roles"][:2]:
        if result["role"].lower() == failing_role:
            assert result["ok"] is False and result["error"]
        else:
            assert result["ok"] is True
            assert result["response"] == "READY [REDACTED] [REDACTED]"
    assert "synthetic-" not in completed.stdout + completed.stderr


def with_key(key_file, *args):
    return subprocess.run(
        [
            sys.executable,
            "examples/with_zen_key.py",
            "--key-file",
            str(key_file),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_explicit_json_key_reaches_both_preflight_and_fitd_without_disclosure(
    endpoint, tmp_path, monkeypatch
):
    key_file = tmp_path / "user key.json"
    key_file.write_text(
        json.dumps({"opencode": {"type": "api", "key": "synthetic-json-secret"}})
    )
    monkeypatch.setenv("TAI_TARGET_API_KEY", "stale-target-key")
    monkeypatch.setenv("TAI_ATTACKER_API_KEY", "stale-attacker-key")
    for command in ("preflight", "fitd"):
        config = profile(endpoint, tmp_path, monkeypatch)
        config.write_text(
            config.read_text()
            .replace(
                'api_key_env = "TAI_TEST_TARGET_KEY"',
                'api_key_env = "TAI_TARGET_API_KEY"',
            )
            .replace(
                'authorization_env = "TAI_TEST_ATTACKER_AUTH"',
                'api_key_env = "TAI_ATTACKER_API_KEY"',
            )
        )
        endpoint.requests.clear()
        if command == "fitd":
            endpoint.scripts["/v1/target/chat/completions"][-1] = Reply(
                "KEY-FOUND\n", echo_authorization=True
            )
        extra = (
            []
            if command == "preflight"
            else [
                "--text",
                "Explain the fictional puzzle",
                "--artifacts-dir",
                str(tmp_path / "runs"),
            ]
        )
        completed = with_key(key_file, command, "--config", str(config), *extra)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert endpoint.requests
        assert all(
            r[1]["Authorization"] == "Bearer synthetic-json-secret"
            for r in endpoint.requests
        )
        assert "synthetic-json-secret" not in completed.stdout + completed.stderr
    _, metadata, events = read_artifacts(tmp_path / "runs")
    assert metadata["settings"]["Target"]["opencode_session"] is True
    assert metadata["settings"]["Attacker"]["opencode_session"] is True
    assert "synthetic-json-secret" not in json.dumps([metadata, events])


@pytest.mark.parametrize(
    "document",
    [
        None,
        b"\xff",
        b'{"opencode": synthetic-json-secret}',
        b"[]",
        b"{}",
        b'{"opencode": []}',
        b'{"opencode": {"type": "oauth", "key": "synthetic-json-secret"}}',
        b'{"opencode": {"type": "api"}}',
        b'{"opencode": {"type": "api", "key": 42}}',
        b'{"opencode": {"type": "api", "key": " "}}',
    ],
)
def test_explicit_key_file_errors_are_safe_and_do_not_send(
    endpoint, tmp_path, document
):
    key_file = tmp_path / "key.json"
    if document is not None:
        key_file.write_bytes(document)
    completed = with_key(key_file, "preflight", "--config", "examples/fitd.toml")
    assert completed.returncode == 2
    assert "opencode" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "synthetic-json-secret" not in completed.stdout + completed.stderr
    assert endpoint.requests == []
