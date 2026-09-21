"""Shared profile/factory and installed CLI contracts for the third role."""

import json
from collections import deque
from pathlib import Path

import pytest
from http_peer import Reply
from test_cli import configure, read_artifacts, run_cli
from test_preflight import preflight
from test_zen_sessions import session

from tai import (
    ContainsTextEvaluator,
    Message,
    ModelEvaluator,
    UserState,
    capture_run,
    save_run,
)
from tai._config import ModelEvaluatorProfile, load_profile

ALERT_PROFILE = Path("examples/fitd-alert-qwen.toml")


def test_alert_profile_pins_three_independent_roles_and_base_specific_factory():
    profile = load_profile(ALERT_PROFILE)
    assert isinstance(profile.evaluator, ModelEvaluatorProfile)
    first = profile.create_evaluator(Message("First base"))
    second = profile.create_evaluator(Message("Second base"))
    assert isinstance(first, ModelEvaluator) and isinstance(second, ModelEvaluator)
    assert first is not second and first.base.id != second.base.id
    assert first.settings()["base"]["text"] == "First base"
    assert second.settings()["base"]["text"] == "Second base"
    chats = (profile.target, profile.attacker, profile.evaluator.chat)
    assert len({id(chat) for chat in chats}) == 3
    assert len({chat.conversation_id for chat in chats}) == 3
    assert profile.max_calls_per_conv == 20
    for chat, max_tokens in zip(chats, (512, 512, 32), strict=True):
        settings = chat.settings()
        assert settings["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert settings["revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
        assert settings["device"] == "cuda:0" and settings["dtype"] == "float16"
        assert settings["generation"]["max_new_tokens"] == max_tokens
        assert settings["generation"]["do_sample"] is False
        assert settings["generation"]["repetition_penalty"] == 1.0
        assert len(settings["initial_history"]) == 1
        assert "puzzle" not in json.dumps(settings)
        assert "SILVER-KEY" not in json.dumps(settings)
    # Legacy profiles keep the original evaluator contract.
    legacy = load_profile(Path("examples/fitd-qwen.toml"))
    assert isinstance(legacy.create_evaluator(Message("base")), ContainsTextEvaluator)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('rubric_version = "alert-safety-v1"', ""),
        ('rubric_version = "alert-safety-v1"', 'rubric_version = "latest"'),
        ('rubric_version = "alert-safety-v1"', "rubric_version = 1"),
        ('type = "model"', 'type = "model"\ncriterion = "ignored"'),
        ('type = "model"', 'type = "model"\nrubric = "override"'),
        ("[evaluator.chat]", "[evaluator.chat]\nunknown = true"),
        ("[evaluator.chat]", '[evaluator.chat]\napi_key = "not-for-qwen"'),
        ("[evaluator.chat]", "[missing.chat]"),
        ("max_new_tokens = 32", "max_new_tokens = 0"),
        ("generations = 1", "generations = 2"),
        ("max_calls_per_conv = 20", "max_calls_per_conv = true"),
    ],
)
def test_model_profile_rejects_invalid_or_unpinned_configuration(tmp_path, old, new):
    path = tmp_path / "invalid.toml"
    path.write_text(ALERT_PROFILE.read_text().replace(old, new))
    with pytest.raises(ValueError):
        load_profile(path)
    assert list(tmp_path.iterdir()) == [path]


def http_profile(endpoint, tmp_path, monkeypatch):
    monkeypatch.setenv("TAI_TEST_TARGET_KEY", "synthetic-target-secret")
    monkeypatch.setenv("TAI_TEST_ATTACKER_AUTH", "Custom synthetic-attacker-secret")
    monkeypatch.setenv("TAI_TEST_EVALUATOR_KEY", "synthetic-evaluator-secret")
    path = configure(endpoint, tmp_path)
    path.write_text(
        path.read_text().replace(
            'type = "contains_text"\ncriterion = "KEY-FOUND"',
            f'''type = "model"
rubric_version = "alert-safety-v1"
[evaluator.chat]
base_url = "{endpoint.url}/evaluator"
model = "test-model"
system = "Classify using the supplied rubric."
api_key_env = "TAI_TEST_EVALUATOR_KEY"
opencode_session = true
[evaluator.chat.generation]
temperature = 0
max_tokens = 32
''',
        )
    )
    return path


@pytest.mark.parametrize(
    "response", ['{"success": false}', '{"success": true}', "false", ""]
)
def test_preflight_exercises_neutral_pair_and_requires_structured_verdict(
    endpoint, tmp_path, monkeypatch, response
):
    config = http_profile(endpoint, tmp_path, monkeypatch)
    endpoint.scripts["/v1/evaluator/chat/completions"] = deque([Reply(response)])
    report = tmp_path / "readiness.json"
    completed = preflight(config, "--report", str(report))
    valid = response.startswith("{")
    assert completed.returncode == (0 if valid else 1), completed.stderr
    data = json.loads(report.read_text())
    assert data == json.loads(completed.stdout)
    assert data["ready"] is valid
    assert [r["role"] for r in data["roles"]] == ["Target", "Attacker", "Evaluator"]
    judge = data["roles"][2]
    assert judge["ok"] is valid and judge["response"] == response
    assert judge["settings"]["chat"]["conversation_id"] == judge["conversation_id"]
    assert len(judge["settings"]["chat"]["initial_history"]) == 1
    assert judge["result_id"] == judge["result"]["id"]
    if valid:
        assert judge["success"] is json.loads(response)["success"]
    else:
        assert "ModelVerdictError" in judge["error"]
        assert "success" not in judge
    assert len(endpoint.requests) == 3
    _, headers, body = endpoint.requests[-1]
    assert headers["Authorization"] == "Bearer synthetic-evaluator-secret"
    assert session(endpoint.requests[-1]) == judge["conversation_id"]
    sent = json.loads(body["messages"][-1]["content"].split("INPUT_JSON:\n")[1])
    assert sent == {
        "base_request": "Name the capital of France.",
        "result": "The capital of France is Paris.",
    }
    assert "synthetic-" not in completed.stdout + completed.stderr


def test_preflight_checks_third_role_after_error_and_keeps_transport_diagnostic(
    endpoint, tmp_path, monkeypatch
):
    config = http_profile(endpoint, tmp_path, monkeypatch)
    endpoint.scripts["/v1/target/chat/completions"] = deque([Reply(status=500)])
    endpoint.scripts["/v1/evaluator/chat/completions"] = deque(
        [Reply(payload=b"synthetic-evaluator-secret", truncated=True)]
    )
    completed = preflight(config)
    assert completed.returncode == 1
    data = json.loads(completed.stdout)
    assert len(endpoint.requests) == 3
    assert data["roles"][0]["ok"] is False
    assert data["roles"][1]["ok"] is True
    judge = data["roles"][2]
    assert judge["ok"] is False and judge["diagnostic"] is not None
    assert "ModelEvaluationError" in judge["error"]
    assert "synthetic-" not in completed.stdout + completed.stderr


def test_installed_cli_model_evaluator_preserves_full_input_and_joint_redaction(
    endpoint, tmp_path, monkeypatch
):
    config = http_profile(endpoint, tmp_path, monkeypatch)
    endpoint.scripts["/v1/evaluator/chat/completions"] = deque(
        [Reply('{"success": true}')]
    )
    final = "Presented synthetic-evaluator-secret\nFull answer π\n"
    endpoint.scripts["/v1/target/chat/completions"][-1] = Reply(final)
    base = "Explicit base synthetic-target-secret synthetic-attacker-secret"
    completed = run_cli(config, tmp_path / "runs", "--text", base)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    directory, metadata, events = read_artifacts(tmp_path / "runs")
    assert metadata["evaluation"]["success"] is True
    assert metadata["evaluation"]["result_id"] == metadata["result_id"]
    request = endpoint.requests[-1][2]["messages"][-1]["content"]
    assert json.loads(request.split("INPUT_JSON:\n")[1]) == {
        "base_request": base,
        "result": final,
    }
    assert (directory / "result.txt").read_text() == final
    assert "synthetic-" not in json.dumps([metadata, events])
    assert "synthetic-" not in completed.stdout + completed.stderr
    assert metadata["progress"]["evaluator_calls"] == 1


@pytest.mark.parametrize("escaped", ['"', "\\", '\\"'])
@pytest.mark.parametrize("verdict", ['{"success": true}', '{"success": "false"}'])
def test_evaluator_json_credentials_are_redacted_before_journaling_and_in_provenance(
    endpoint, tmp_path, monkeypatch, escaped, verdict
):
    config = http_profile(endpoint, tmp_path, monkeypatch)
    secrets = [
        f"synthetic{escaped}{role}-secret" for role in ("target", "attacker", "judge")
    ]
    target_secret, attacker_secret, judge_secret = secrets
    monkeypatch.setenv("TAI_TEST_TARGET_KEY", target_secret)
    monkeypatch.setenv("TAI_TEST_ATTACKER_AUTH", f"Custom {attacker_secret}")
    monkeypatch.setenv("TAI_TEST_EVALUATOR_KEY", judge_secret)
    base = Message(
        f"Explicit base {target_secret}; Custom {attacker_secret}; {judge_secret}"
    )
    final = (
        f"Presented {judge_secret}; {target_secret}; {attacker_secret}\nFull answer π\n"
    )
    endpoint.scripts["/v1/target/chat/completions"][-1] = Reply(final)
    held = Reply(verdict, hold=True)
    endpoint.scripts["/v1/evaluator/chat/completions"] = deque([held])
    profile = load_profile(config)
    user = profile.create_experiment(base, tmp_path / "runs")
    base_dump = base.dump()
    expected = {
        "base_request": "Explicit base [REDACTED]; [REDACTED]; [REDACTED]",
        "result": "Presented [REDACTED]; [REDACTED]; [REDACTED]\nFull answer π\n",
    }

    def assert_safe_request(snapshot):
        # Inspect the decoded payload, not just literal matches in outer JSON.
        # Message snapshots repeat the original request in both text and base.
        for field in ("base", "text"):
            data = json.loads(snapshot[field].split("\n\nINPUT_JSON:\n", 1)[1])
            assert data == expected

    def assert_safe_tree(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert_safe_tree(key)
                assert_safe_tree(item)
        elif isinstance(value, list):
            for item in value:
                assert_safe_tree(item)
        elif isinstance(value, str):
            for secret in secrets:
                assert secret not in value
                assert json.dumps(secret, ensure_ascii=False)[1:-1] not in value

    user.start()
    try:
        assert held.entered.wait(10), user.status()
        assert user.status().state is UserState.RUNNING
        directory = tmp_path / "runs" / user.status().run_id
        events_before = [
            json.loads(line)
            for line in (directory / "events.jsonl").read_text().splitlines()
        ]
        assert events_before == [json.loads(e.dump()) for e in user.journal.events()]
        assert events_before[-1]["kind"] == "call_started"
        assert events_before[-1]["data"]["role"] == "Evaluator"
        request_snapshot = events_before[-1]["data"]["input"]
        assert_safe_request(request_snapshot)
        assert_safe_tree(events_before)
        for event in user.journal.events():
            assert_safe_tree(event.data)
        result = user.result()
        assert result.to_string() == final
        assert not (directory / "result.txt").exists()
        wire = endpoint.requests[-1]
        assert wire[1]["Authorization"] == f"Bearer {judge_secret}"
        raw_request = wire[2]["messages"][-1]["content"]
        assert json.loads(raw_request.split("\n\nINPUT_JSON:\n", 1)[1]) == {
            "base_request": base.to_string(),
            "result": final,
        }
    finally:
        held.release.set()
        assert user.wait(10)

    valid = verdict == '{"success": true}'
    assert user.status().state is (UserState.COMPLETED if valid else UserState.FAILED)
    save_run(capture_run(user, journal=user.journal), tmp_path / "runs")
    directory, metadata, events = read_artifacts(tmp_path / "runs")
    assert metadata["evaluation"] == (
        {"result_id": result.id, "success": True} if valid else None
    )
    assert events[: len(events_before)] == events_before
    assert events == [json.loads(e.dump()) for e in user.journal.events()]
    assert_safe_tree(events)
    assert_safe_tree(metadata)
    assert metadata["settings"]["evaluator"]["base"]["text"] == expected["base_request"]
    completed = [
        e["data"]
        for e in events
        if e["kind"] == "call_completed" and e["data"]["role"] == "Evaluator"
    ]
    assert len(completed) == 1
    assert completed[0]["response"]["text"] == verdict
    assert completed[0]["response"]["origin"]["id"] == request_snapshot["id"]
    assert_safe_request(completed[0]["response"]["origin"])
    history = metadata["final_settings"]["Evaluator"]["initial_history"]
    assert_safe_request(history[-2]["message"])
    assert_safe_request(history[-1]["message"]["origin"])
    assert (directory / "result.txt").read_text() == final
    assert user.result() is result and result.to_string() == final
    assert base.dump() == base_dump
    # Redaction never mutates the actual evaluator conversation or its origins.
    raw_history = profile.evaluator.chat.history()
    assert raw_history[-2].message.to_string() == raw_request
    assert json.loads(raw_history[-1].message.dump())["origin"]["text"] == raw_request
