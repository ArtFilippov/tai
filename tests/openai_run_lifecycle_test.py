"""Interrupted responses and independent repeated HTTP-backed FITD runs."""

import json
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pytest
from chats_assertions import HttpEndpoint, object_at
from http_peer import Reply
from openai_fitd_support import accepted_user, artifacts, finish_run, puzzle_user, result_file
from openai_workers import observe_partial_worker

import tai
from tai import ChatTurn, Message, OpenAIChat


def _assert_interrupted_state(
    user: tai.FITDExperiment, chats: dict[str, OpenAIChat], endpoint: HttpEndpoint,
    result: Message, evaluation: object, histories: dict[str, tuple[ChatTurn, ...]], calls: int,
) -> None:
    assert user.status().state is tai.UserState.FAILED
    assert user.result() is result
    assert result.to_string() == "Accepted report"
    assert user.status().evaluation == evaluation
    actual_evaluation = user.status().evaluation
    assert actual_evaluation is not None
    assert actual_evaluation.result_id == result.id and actual_evaluation.success is False
    assert {role: chat.history() for role, chat in chats.items()} == histories
    assert len(endpoint.requests) == calls
    assert user.status().progress.target_calls == 5


def _assert_interrupted_artifacts(
    user: tai.FITDExperiment, root: Path, stop: str, prefix: bytes, result: Message,
) -> None:
    events, metadata = artifacts(user, root)
    kind = "call_cancelled" if stop == "cancel" else "call_failed"
    interrupted = next(event["data"] for event in events if event["kind"] == kind)
    reason = user.status().reason
    assert reason is not None
    assert interrupted["diagnostic"] == {
        "status_code": 200, "response_bytes": len(prefix),
        "response_fragment": "OBSERVED_PARTIAL KEY-FOUND [REDACTED]",
    }
    if stop == "cancel":
        assert interrupted["response"] is None
        assert "ChatCancelled" in reason
    else:
        assert interrupted["error"] == user.status().reason
        assert "OpenAITimeoutError" in reason
    starts = [event["data"] for event in events if event["kind"] == "call_started"]
    assert interrupted["call_id"] == starts[-1]["call_id"]
    assert interrupted["role"] == "Target"
    assert not any(
        event["kind"] == "call_completed" and event["data"]["call_id"] == interrupted["call_id"]
        for event in events
    )
    assert "UNSEEN_SUFFIX" not in json.dumps(events)
    assert metadata["result"] == json.loads(result.dump())
    assert metadata["evaluation"] == {"result_id": result.id, "success": False}
    assert result_file(root, user).read_text(encoding="utf-8") == "Accepted report"


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_real_fitd_interrupted_body_is_only_a_diagnostic(
    endpoint: HttpEndpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str,
) -> None:
    prefix = b"OBSERVED_PARTIAL KEY-FOUND synthetic-target-secret"
    observed, workers = observe_partial_worker(monkeypatch, prefix)
    held = Reply(payload=prefix + b"UNSEEN_SUFFIX", hold=True, partial=True, partial_bytes=len(prefix))
    user, chats = accepted_user(endpoint, tmp_path, held, timeout=1 if stop == "timeout" else 120)
    user.start()
    with ExitStack() as cleanup:
        cleanup.callback(finish_run, user, held)
        assert held.entered.wait(5)
        assert observed.wait(5), "worker did not report the received body prefix"
        result = user.result()
        assert result is not None
        evaluation = user.status().evaluation
        histories = {role: chat.history() for role, chat in chats.items()}
        calls = len(endpoint.requests)
        if stop == "cancel":
            user.stop()
            # Exchange cancellation remains failure even after a pending User.stop.
            chats["Target"].cancel()
        assert user.wait(3)
        assert held.disconnected.wait(1)
        assert not held.release.is_set()
        _assert_interrupted_state(user, chats, endpoint, result, evaluation, histories, calls)
        _assert_interrupted_artifacts(user, tmp_path, stop, prefix, result)
        workers.assert_reaped()


@dataclass(frozen=True)
class PreviousRun:
    result: Message
    dump: str
    directory: Path
    metadata: dict[str, object]
    files: dict[str, bytes]
    points: dict[str, object]
    offset: int


def _first_run(
    user: tai.FITDExperiment, endpoint: HttpEndpoint, first_outcome: str,
) -> None:
    held = Reply(hold=True)
    if first_outcome == "STOPPED":
        endpoint.scripts["/v1/target/chat/completions"][3] = held
    user.start()
    with ExitStack() as cleanup:
        cleanup.callback(finish_run, user, held)
        if first_outcome == "STOPPED":
            assert held.entered.wait(5)
            user.stop()
            assert not user.wait(0)
            held.release.set()
        assert user.wait(10)
        assert user.status().state.value == first_outcome, user.status().reason


def _snapshot(
    user: tai.FITDExperiment, chats: dict[str, OpenAIChat], root: Path, endpoint: HttpEndpoint,
) -> PreviousRun:
    result = user.result()
    assert result is not None
    directory = result_file(root, user).parent
    _, metadata = artifacts(user, root)
    return PreviousRun(
        result, result.dump(), directory, metadata,
        {path.name: path.read_bytes() for path in directory.iterdir()},
        {role: chat.checkpoint() for role, chat in chats.items()}, len(endpoint.requests),
    )


def _restart_scripts(endpoint: HttpEndpoint, next_initial: Reply) -> None:
    endpoint.scripts = {
        "/v1/attacker/chat/completions": deque(
            [next_initial] + [Reply(text) for text in ["L1", "L2", "L3"] + ["yes"] * 5]
        ),
        "/v1/target/chat/completions": deque(Reply(text) for text in (
            "New one", "New two", "New three", "New four", "KEY-FOUND new solution",
        )),
    }


def _assert_restart_admitted(
    user: tai.FITDExperiment, chats: dict[str, OpenAIChat], previous: PreviousRun, root: Path,
) -> None:
    assert user.status().run_id != previous.directory.name
    assert user.result() is None
    assert user.status().evaluation is None
    assert all(len(chat.history()) == 1 for chat in chats.values())
    with pytest.raises(ValueError, match="checkpoint"):
        chats["Target"].restore(previous.points["Target"])
    # Run-entry reset and Attacker reset retain both ownership reservations.
    contender = tai.FITDExperiment(
        chats["Target"], tai.LLMChat(lambda turns: "unused"), Message("Other"),
        tai.ContainsTextEvaluator("KEY-FOUND"), journal_dir=root,
    )
    with pytest.raises(RuntimeError, match="LLMChat.*active"):
        contender.start()


def _assert_repeat_artifacts(
    user: tai.FITDExperiment, previous: PreviousRun, root: Path,
) -> None:
    assert user.status().state is tai.UserState.COMPLETED, user.status().reason
    assert user.status().reason == "evaluator_success"
    result = user.result()
    assert result is not None and result.to_string() == "KEY-FOUND new solution"
    _, metadata = artifacts(user, root)
    for role in ("Target", "Attacker"):
        assert object_at(metadata, "settings", role).pop("conversation_id") != object_at(
            previous.metadata, "settings", role,
        ).pop("conversation_id")
    assert metadata["settings"] == previous.metadata["settings"]
    assert previous.result.dump() == previous.dump
    assert {
        path.name: path.read_bytes() for path in previous.directory.iterdir()
    } == previous.files


def _assert_repeat_requests(
    endpoint: HttpEndpoint, chats: dict[str, OpenAIChat], previous: PreviousRun,
) -> None:
    for role, chat in chats.items():
        before = chat.history()
        with pytest.raises(ValueError, match="checkpoint"):
            chat.restore(previous.points[role])
        assert chat.history() == before
        first = next(
            request for request in endpoint.requests[previous.offset:]
            if f"/{role.lower()}/" in request[0]
        )
        assert [turn["role"] for turn in first[2]["messages"]] == ["system", "user"]
        assert first[2]["messages"][0]["content"] == (
            "Target puzzle keeper" if role == "Target" else "Attacker puzzle author"
        )


def _finish_restart(user: tai.FITDExperiment, next_initial: Reply) -> None:
    next_initial.release.set()
    assert user.wait(10)


@pytest.mark.parametrize("first_outcome", ["COMPLETED", "STOPPED"])
def test_same_user_restarts_actual_http_chats_with_fresh_history_and_safe_settings(
    endpoint: HttpEndpoint, tmp_path: Path, first_outcome: str,
) -> None:
    user, chats = puzzle_user(endpoint, tmp_path)
    _first_run(user, endpoint, first_outcome)
    previous = _snapshot(user, chats, tmp_path, endpoint)
    next_initial = Reply("New opening clue", hold=True)
    _restart_scripts(endpoint, next_initial)
    user.start()
    with ExitStack() as cleanup:
        cleanup.callback(_finish_restart, user, next_initial)
        assert next_initial.entered.wait(5)
        _assert_restart_admitted(user, chats, previous, tmp_path)
    _assert_repeat_artifacts(user, previous, tmp_path)
    _assert_repeat_requests(endpoint, chats, previous)
