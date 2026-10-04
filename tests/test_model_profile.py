"""Shared profile/factory and installed CLI contracts for the third role."""

import json
from collections import deque
from contextlib import closing
from pathlib import Path

import pytest
from app_test_support import HttpPeer, session
from app_redaction_support import assert_saved, capture_live, make_case
from http_peer import Reply
from test_cli import configure, read_artifacts, run_cli
from test_preflight import preflight

from tai import (
    ContainsTextEvaluator,
    Message,
    ModelEvaluator,
    QwenChat,
)
from alert.values import array_value, object_value
from app.configuration import load_profile

ALERT_PROFILE = Path("examples/fitd-alert-qwen.toml")
MIXED_PROFILE = Path("examples/fitd-alert-qwen3-4b-transformers.toml")


def test_mixed_alert_profile_pins_two_shared_weight_groups_without_loading() -> None:
    with closing(load_profile(MIXED_PROFILE)) as profile, closing(profile.fresh()) as fresh:
        assert profile.evaluator.chat is not None
        chats = (profile.target, profile.attacker, profile.evaluator.chat,
                 fresh.target, fresh.attacker)
        target, attacker, judge, next_target, next_attacker = chats
        assert isinstance(target, QwenChat)
        assert isinstance(attacker, QwenChat)
        assert isinstance(judge, QwenChat)
        assert isinstance(next_target, QwenChat)
        assert isinstance(next_attacker, QwenChat)
        assert profile.max_calls_per_conv == 10
        assert target._executor is judge._executor is next_target._executor
        assert attacker._executor is next_attacker._executor
        assert attacker._executor is not target._executor
        for chat, model, revision, tokens in (
            (target, "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", 512),
            (attacker, "Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", 512),
            (judge, "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", 320),
        ):
            settings = object_value(chat.settings(), "settings")
            assert (settings["model"], settings["revision"]) == (model, revision)
            assert settings["backend"] == "qwen" and settings["cache_dir"] == "/vg/mnt/titan"
            assert settings["device"] == "cuda:0" and settings["dtype"] == "float16"
            assert settings["loaded"] is False and settings["actual_dtype"] is None
            assert chat.history() == ()
            generation = object_value(settings["generation"], "generation")
            assert generation["max_new_tokens"] == tokens
            assert generation["do_sample"] is False and generation["repetition_penalty"] == 1.0


def test_alert_profile_pins_three_independent_roles_and_reusable_evaluator() -> None:
    profile = load_profile(ALERT_PROFILE)
    assert isinstance(profile.evaluator.evaluator, ModelEvaluator)
    first = profile.create_experiment(Message("First base"))
    second = profile.create_experiment(Message("Second base"))
    assert first._settings.evaluator is second._settings.evaluator is profile.evaluator.evaluator
    assert first.base.to_string() == "First base"
    assert second.base.to_string() == "Second base"
    assert "base" not in profile.evaluator.settings()
    chats = (profile.target, profile.attacker, profile.evaluator.chat)
    assert profile.evaluator.chat is not None
    assert all(chat is not None for chat in chats)
    assert len({id(chat) for chat in chats}) == 3
    assert len({chat.conversation_id for chat in chats if chat is not None}) == 3
    assert profile.max_calls_per_conv == 20
    for chat, max_tokens in zip(chats, (512, 512, 32), strict=True):
        assert chat is not None
        settings = object_value(chat.settings(), "settings")
        assert settings["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert settings["revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
        assert settings["device"] == "cuda:0" and settings["dtype"] == "float16"
        generation = object_value(settings["generation"], "generation")
        assert generation["max_new_tokens"] == max_tokens
        assert generation["do_sample"] is False
        assert generation["repetition_penalty"] == 1.0
        assert array_value(settings["initial_history"], "history") == []
        assert "puzzle" not in json.dumps(settings)
        assert "SILVER-KEY" not in json.dumps(settings)
    text_profile = load_profile(Path("examples/fitd-qwen.toml"))
    assert isinstance(text_profile.evaluator.evaluator, ContainsTextEvaluator)


@pytest.mark.parametrize("name", ["qwen", "transformers", "vllm"])
@pytest.mark.parametrize("custom_system", [False, True])
def test_alert_profiles_reset_and_recreate_only_explicit_system_instructions(
    tmp_path: Path, name: str, custom_system: bool,
) -> None:
    path = Path(f"examples/fitd-alert-{name}.toml")
    if custom_system:
        source = path.read_text(encoding="utf-8")
        for role in ("target", "attacker", "evaluator.chat"):
            source = source.replace(f"[{role}]", f'[{role}]\nsystem = "Custom {role}"')
        path = tmp_path / "custom.toml"
        path.write_text(source, encoding="utf-8")
    profile = load_profile(path)
    fresh = profile.fresh()
    chats = (profile.target, profile.attacker, profile.evaluator.chat,
             fresh.target, fresh.attacker, fresh.evaluator.chat)
    assert len({id(chat) for chat in chats}) == 6
    assert profile.evaluator.evaluator is not fresh.evaluator.evaluator
    conversations: set[str] = set()
    for chat, role in zip(chats, ("target", "attacker", "evaluator.chat") * 2, strict=True):
        assert chat is not None
        expected = [("system", f"Custom {role}")] if custom_system else []
        assert [(turn.role, turn.text()) for turn in chat.history()] == expected
        conversations.add(chat.conversation_id)
        chat.reset()
        assert [(turn.role, turn.text()) for turn in chat.history()] == expected
        conversations.add(chat.conversation_id)
        settings = object_value(chat.settings(), "settings")
        assert len(array_value(settings["initial_history"], "history")) == len(expected)
    assert len(conversations) == 12
    profile.close()


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
def test_model_profile_rejects_invalid_or_unpinned_configuration(tmp_path: Path, old: str, new: str) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(ALERT_PROFILE.read_text(encoding="utf-8").replace(old, new))
    with pytest.raises(ValueError):
        load_profile(path)
    assert list(tmp_path.iterdir()) == [path]


def http_profile(endpoint: HttpPeer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    endpoint: HttpPeer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: str,
) -> None:
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
    assert judge["exchange_settings"]["conversation_id"] == judge["conversation_id"]
    assert len(judge["exchange_settings"]["initial_history"]) == 1
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
    endpoint: HttpPeer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    endpoint: HttpPeer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = http_profile(endpoint, tmp_path, monkeypatch)
    endpoint.scripts["/v1/evaluator/chat/completions"] = deque(
        [Reply('{"success": true}')]
    )
    final = "Presented synthetic-evaluator-secret\nFull answer π\n"
    endpoint.scripts["/v1/target/chat/completions"][-1] = Reply(final)
    base = "Explicit base synthetic-target-secret synthetic-attacker-secret"
    completed = run_cli(config, tmp_path / "runs", "--text", base)
    assert not completed.returncode, completed.stdout + completed.stderr
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
    endpoint: HttpPeer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, escaped: str, verdict: str,
) -> None:
    config = http_profile(endpoint, tmp_path, monkeypatch)
    case, held = make_case(endpoint, config, monkeypatch, escaped, verdict)
    base_dump = case.base.dump()
    case.user.start()
    try:
        evidence = capture_live(case, endpoint, held)
    finally:
        held.release.set()
        assert case.user.wait(10)
    assert_saved(case, evidence, verdict, base_dump)
