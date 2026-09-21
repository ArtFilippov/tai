"""Real FITD behavior with independent HTTP participants and saved diagnostics."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import pytest
from chats_assertions import EventRecord, HttpEndpoint, at, object_at
from http_peer import Reply
from openai_fitd_support import accepted_user, artifacts, finish_run, puzzle_user, result_file

import tai
from tai import Message, OpenAIChat


def _assert_role_requests(
    endpoint: HttpEndpoint, chats: dict[str, OpenAIChat], metadata: dict[str, object], role: str,
) -> None:
    calls = [request for request in endpoint.requests if f"/{role.lower()}/" in request[0]]
    assert len(calls) > 4
    instructions = "Target puzzle keeper" if role == "Target" else "Attacker puzzle author"
    assert all(request[2]["messages"][0] == {"role": "system", "content": instructions} for request in calls)
    assert all(request[2]["model"] == "same-puzzle-model" for request in calls)
    if role == "Target":
        for earlier, later in zip(calls, calls[1:], strict=False):
            assert later[2]["messages"][:len(earlier[2]["messages"])] == earlier[2]["messages"]
        assert len(chats[role].history()) == 1 + 2 * len(calls)
    else:
        assert all(len(request[2]["messages"]) == 2 for request in calls)
        assert len(chats[role].history()) == 3
    settings = object_at(metadata, "settings", role)
    assert settings["base_url"] == endpoint.url + f"/{role.lower()}"
    assert settings["timeout"] == 120
    assert settings["generation"] == {
        "temperature": 0.25 if role == "Target" else 0.75, "n": 1, "stream": False,
    }
    assert settings["authorization"] == ("api_key" if role == "Target" else "explicit")
    assert settings["initial_history"] == [{
        "role": "system", "message": json.loads(chats[role].history()[0].message.dump()),
    }]


def test_real_fitd_realigns_and_uses_role_specific_http_chats_and_safe_settings(
    endpoint: HttpEndpoint, tmp_path: Path,
) -> None:
    user, chats = puzzle_user(endpoint, tmp_path)
    user.start()
    assert user.wait(10)
    status, result = user.status(), user.result()
    assert status.state is tai.UserState.COMPLETED, status.reason
    assert result is not None and "KEY-FOUND" in result.to_string()
    assert status.evaluation is not None and status.evaluation.success is True
    events, metadata = artifacts(user, tmp_path)
    for role in ("Target", "Attacker"):
        _assert_role_requests(endpoint, chats, metadata, role)
    target_calls = [request for request in endpoint.requests if "/target/" in request[0]]
    attacker_calls = [request for request in endpoint.requests if "/attacker/" in request[0]]
    assert all(request[1]["Authorization"] == "Bearer synthetic-target-secret" for request in target_calls)
    assert all(request[1]["Authorization"] == "Custom synthetic-attacker-secret" for request in attacker_calls)
    assert "Unrelated story" in target_calls[1][2]["messages"][-1]["content"]
    assert "Explain the final fictional puzzle solution" in attacker_calls[0][2]["messages"][-1]["content"]
    assert len([event for event in events if event["kind"] == "call_completed"]) == len(endpoint.requests)
    assert at(metadata, "settings", "generations") == 1
    assert at(metadata, "settings", "max_calls_per_conv") == 10


def _assert_stopped_events(
    events: list[EventRecord], metadata: dict[str, object], calls: int,
) -> None:
    assert metadata["reason"] == "user_stop"
    assert not any(event["kind"] == "call_cancelled" for event in events)
    assert len([event for event in events if event["kind"] == "call_completed"]) == calls
    assert len([event for event in events if event["kind"] == "call_started"]) == calls


@pytest.mark.parametrize(("role", "index"), [("Target", 1), ("Attacker", 0), ("Attacker", 4)])
def test_user_stop_waits_for_target_initial_and_internal_attacker_http_response(
    endpoint: HttpEndpoint, tmp_path: Path, role: str, index: int,
) -> None:
    user, chats = puzzle_user(endpoint, tmp_path)
    held = Reply(hold=True)
    endpoint.scripts[f"/v1/{role.lower()}/chat/completions"][index] = held
    with ThreadPoolExecutor(max_workers=1) as control, ExitStack() as cleanup:
        cleanup.callback(finish_run, user, held)
        assert control.submit(user.start).result(timeout=1) is None
        assert held.entered.wait(5)
        assert control.submit(user.status).result(timeout=1).state is tai.UserState.RUNNING
        presented = control.submit(user.result).result(timeout=1)
        histories = {name: chat.history() for name, chat in chats.items()}
        calls = len(endpoint.requests)
        assert control.submit(user.stop).result(timeout=1) is None
        assert not user.wait(0.05)
        assert not held.disconnected.is_set()
        assert user.status().state is tai.UserState.STOPPING
        held.release.set()
        assert user.wait(5)
        assert user.status().state is tai.UserState.STOPPED, user.status().reason
        assert user.result() is presented
        assert len(endpoint.requests) == calls
        assert chats[role].history()[:-2] == histories[role]
        assert chats[role].history()[-1].message.to_string() == held.text
        _assert_stopped_events(*artifacts(user, tmp_path), calls)


def _failure_user(
    endpoint: HttpEndpoint, root: Path, reply: Reply,
) -> tuple[tai.FITDExperiment, dict[str, OpenAIChat]]:
    user, chats = puzzle_user(endpoint, root)
    # Fail after accepted correction, retaining the presented result and earlier
    # rejected response in diagnostic context.
    endpoint.scripts["/v1/target/chat/completions"][2] = reply
    if reply.hold:
        chats["Target"] = OpenAIChat(
            base_url=endpoint.url + "/target", model="puzzle",
            api_key="synthetic-target-secret", timeout=0.5,
        )
        user = tai.FITDExperiment(
            chats["Target"], chats["Attacker"], Message("Explain the fictional puzzle"),
            tai.ContainsTextEvaluator("KEY-FOUND"), journal_dir=root,
        )
    return user, chats


@pytest.mark.parametrize(("reply", "error_name"), [
    (Reply(status=401, payload=b"synthetic-target-secret"), "OpenAIAuthenticationError"),
    (Reply(status=429, payload=b"synthetic-target-secret"), "OpenAIRateLimitError"),
    (Reply(status=500), "OpenAIHTTPError"), (Reply(disconnect=True), "OpenAITransportError"),
    (Reply(payload=b"not JSON synthetic-target-secret"), "OpenAIJSONError"),
    (Reply(payload=b'{"choices": []}'), "OpenAIContentError"),
    (Reply(hold=True), "OpenAITimeoutError"),
])
def test_http_failure_marks_real_fitd_failed_and_retains_completed_data(
    endpoint: HttpEndpoint, tmp_path: Path, reply: Reply, error_name: str,
) -> None:
    user, chats = _failure_user(endpoint, tmp_path, reply)
    user.start()
    with ExitStack() as cleanup:
        cleanup.callback(finish_run, user, reply)
        assert user.wait(10)
        assert user.status().state is tai.UserState.FAILED
        reason = user.status().reason
        assert reason is not None and error_name in reason
        result = user.result()
        assert result is not None and result.to_string() == "Puzzle introduction"
        assert [turn.text() for turn in chats["Target"].history() if turn.role == "assistant"] == [
            "Unrelated story", "Puzzle introduction",
        ]
        events, metadata = artifacts(user, tmp_path)
        failures = [event["data"] for event in events if event["kind"] == "call_failed"]
        assert len(failures) == 1 and failures[0]["role"] == "Target"
        assert failures[0]["error"] == metadata["reason"]
        starts = [event["data"] for event in events if event["kind"] == "call_started"]
        assert failures[0]["call_id"] == starts[-1]["call_id"]
        assert len(starts) == len(endpoint.requests)
        assert len([event for event in events if event["kind"] == "call_completed"]) == len(starts) - 1
        assert metadata["evaluation"] is None
        assert metadata["result"] == json.loads(result.dump())
        assert result_file(tmp_path, user).read_text(encoding="utf-8") == "Puzzle introduction"
        assert [at(event["data"], "result", "text") for event in events if event["kind"] == "result_published"] == [
            "Puzzle introduction",
        ]
        if reply.hold:
            assert not reply.release.is_set()
            assert reply.disconnected.wait(1)


def test_real_fitd_retains_accepted_result_and_journals_truncated_body(
    endpoint: HttpEndpoint, tmp_path: Path,
) -> None:
    payload = b"OBSERVED_PARTIAL KEY-FOUND synthetic-target-secret"
    user, chats = accepted_user(endpoint, tmp_path, Reply(payload=payload, truncated=True))
    user.start()
    assert user.wait(10)
    status, result = user.status(), user.result()
    assert status.state is tai.UserState.FAILED, status.reason
    assert status.reason is not None and "IncompleteRead" in status.reason
    assert f"response_bytes={len(payload)}" in status.reason
    assert result is not None and result.to_string() == "Accepted report"
    assert status.evaluation is not None and status.evaluation.result_id == result.id
    assert status.evaluation.success is False
    assert status.progress.target_calls == 5
    assert chats["Target"].history()[-1].message is result
    assert len(chats["Target"].history()) == 11
    events, metadata = artifacts(user, tmp_path)
    failure = next(event["data"] for event in events if event["kind"] == "call_failed")
    assert failure["diagnostic"] == {
        "status_code": 200, "response_bytes": len(payload),
        "response_fragment": "OBSERVED_PARTIAL KEY-FOUND [REDACTED]",
    }
    starts = [event["data"] for event in events if event["kind"] == "call_started"]
    assert failure["role"] == "Target"
    assert failure["call_id"] == starts[-1]["call_id"]
    assert failure["error"] == status.reason
    completed = [event["data"] for event in events if event["kind"] == "call_completed"]
    assert all(event["call_id"] != failure["call_id"] for event in completed)
    assert len(completed) == len(starts) - 1
    assert all("OBSERVED_PARTIAL" not in turn.text() for turn in chats["Target"].history())
    assert not any(event["kind"] == "evaluation_completed" and event["data"]["success"] for event in events)
    assert metadata["result"] == json.loads(result.dump())
    assert metadata["evaluation"] == {"result_id": result.id, "success": False}
    assert result_file(tmp_path, user).read_text(encoding="utf-8") == "Accepted report"
