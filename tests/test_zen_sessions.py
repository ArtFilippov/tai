"""Conversation identity on the wire, including real FITD adaptation."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from uuid import UUID

import pytest
from http_peer import Reply
from chats_assertions import HttpEndpoint, RequestRecord, at, text_at

from tai import (
    ChatCancelled,
    ContainsTextEvaluator,
    Message,
    OpenAIChat,
    OpenAIHTTPError,
    UserState,
    capture_run,
    save_run,
)
from tai import (
    FITDExperiment as FITDUser,
)


def session(request: RequestRecord) -> str | None:
    return {key.lower(): value for key, value in request[1].items()}.get(
        "x-opencode-session"
    )


def _assert_authorization(endpoint: HttpEndpoint) -> None:
    for _, headers, _ in endpoint.requests:
        assert headers["User-Agent"] == f"tai/{version('tai')}"
        assert headers["Authorization"] == "Bearer synthetic-session-secret"


def _assert_session_settings(chat: OpenAIChat, identity: str) -> None:
    assert at(chat.settings(), "conversation_id") == identity
    assert at(chat.settings(), "opencode_session") is True


def test_session_follows_conversation_through_retry_rollback_and_reset(endpoint: HttpEndpoint) -> None:
    endpoint.require_session = True
    ordinary = OpenAIChat(base_url=endpoint.url, model="puzzle")
    with pytest.raises(OpenAIHTTPError, match="HTTP 400"):
        ordinary.send(Message("No session configured"))
    assert session(endpoint.requests.pop()) is None
    endpoint.scripts["/v1/chat/completions"] = deque(
        [
            Reply("First"),
            Reply(status=503),
            Reply("Retry"),
            Reply("Branch"),
            Reply("New"),
        ]
    )
    chat = OpenAIChat(
        base_url=endpoint.url,
        model="puzzle",
        opencode_session=True,
        api_key="synthetic-session-secret",
    )
    initial_id = chat.conversation_id
    UUID(initial_id)
    assert initial_id != ordinary.conversation_id
    chat.send(Message("First"))
    point = chat.checkpoint()
    with pytest.raises(OpenAIHTTPError):
        chat.send(Message("Failed"))
    assert chat.conversation_id == initial_id
    chat.send(Message("Retry"))
    chat.restore(point)
    chat.send(Message("Branch"))
    assert [session(r) for r in endpoint.requests] == [initial_id] * 4
    assert "Retry" not in str(endpoint.requests[-1][2]["messages"])
    chat.reset()
    new_id = chat.conversation_id
    assert new_id != initial_id
    with pytest.raises(ValueError):
        chat.restore(point)
    chat.send(Message("New"))
    assert session(endpoint.requests[-1]) == new_id
    _assert_session_settings(chat, new_id)
    _assert_authorization(endpoint)


def test_failed_reset_during_send_and_cancellation_preserve_session(endpoint: HttpEndpoint) -> None:
    endpoint.require_session = True
    held = Reply(hold=True)
    endpoint.scripts["/v1/chat/completions"] = deque([held, Reply("Retry")])
    chat = OpenAIChat(base_url=endpoint.url, model="puzzle", opencode_session=True)
    initial_id = chat.conversation_id
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(chat.send, Message("Held"))
        with ExitStack() as cleanup:
            cleanup.callback(held.release.set)
            cleanup.callback(chat.cancel)
            assert held.entered.wait(3)
            with pytest.raises(RuntimeError):
                chat.reset()
            assert chat.conversation_id == initial_id
            chat.cancel()
            with pytest.raises(ChatCancelled):
                pending.result(timeout=3)
    assert chat.conversation_id == initial_id
    chat.send(Message("Retry"))
    assert [session(r) for r in endpoint.requests] == [initial_id] * 2


def _scripts(endpoint: HttpEndpoint, branch: str) -> None:
    target = ["Opening", "I cannot help with that", "Gentle", "Clue", "Draft", "KEY-FOUND"]
    attacker = ["Initial", "L1", "L2", "L3", "yes", "Rephrased"] + ["yes"] * 5
    if branch == "restart":
        target[:2] = ["I cannot help with that", "Opening"]
        attacker = ["Initial", "L1", "L2", "L3", "Safer"] + ["yes"] * 6
    endpoint.scripts = {
        "/v1/target/chat/completions": deque(Reply(text) for text in target),
        "/v1/attacker/chat/completions": deque(Reply(text) for text in attacker),
    }


def _assert_adaptation(calls: list[RequestRecord], branch: str) -> None:
    targets = [r for r in calls if "/target/" in r[0]]
    attackers = [r for r in calls if "/attacker/" in r[0]]
    target_ids = [session(r) for r in targets]
    attacker_ids = [session(r) for r in attackers]
    assert len(targets) == 6 and len(attackers) > 6  # Includes internal checks.
    assert len(set(attacker_ids)) == len(attackers)
    assert set(target_ids).isdisjoint(attacker_ids)
    if branch == "restart":
        assert target_ids[0] != target_ids[1]
        assert len(set(target_ids[1:])) == 1
        assert len(targets[1][2]["messages"]) == 2
    else:
        assert len(set(target_ids)) == 1
        assert targets[2][2]["messages"][:-1] == targets[1][2]["messages"][:-1]
    for _, headers, _ in calls:
        assert headers["User-Agent"] == f"tai/{version('tai')}"
    for role, role_calls in (("target", targets), ("attacker", attackers)):
        assert all(r[1]["Authorization"] == f"Bearer synthetic-{role}-secret" for r in role_calls)


def _assert_journal(
    user: FITDUser, chats: list[OpenAIChat], calls: list[RequestRecord], branch: str,
) -> None:
    events = user.journal.events()
    assert any(e.kind == "branch_selected" and at(e.data, "branch") == branch for e in events)
    started = [e.data for e in events if e.kind == "call_started"]
    assert [at(e, "conversation_id") for e in started] == [session(r) for r in calls]
    resets = [
        e.data for e in events if e.kind == "history_reset"
        and at(e.data, "context", "cause", "reason") != "run_started"
    ]
    assert {text_at(e, "conversation_id") for e in resets} <= {session(r) for r in calls} | {
        c.conversation_id for c in chats
    }


@pytest.mark.parametrize("branch", ["restart", "rephrase"])
def test_real_fitd_sessions_follow_adaptation_and_independent_repeated_runs(
    endpoint: HttpEndpoint, tmp_path: Path, branch: str,
) -> None:
    endpoint.require_session = True
    chats = [
        OpenAIChat(
            base_url=endpoint.url + f"/{role}", model="same-puzzle-model",
            system=f"{role} puzzle rules", api_key=f"synthetic-{role}-secret",
            opencode_session=True,
        ) for role in ("target", "attacker")
    ]
    user = FITDUser(
        chats[0], chats[1], Message("Explain the final fictional puzzle solution"),
        ContainsTextEvaluator("KEY-FOUND"), journal_dir=tmp_path, max_calls_per_conv=6,
    )
    previous_ids: set[str | None] = {chat.conversation_id for chat in chats}
    for _ in range(2):
        _scripts(endpoint, branch)
        endpoint.requests.clear()
        user.start()
        assert user.wait(10)
        directory = save_run(capture_run(user, journal=user.journal), tmp_path)
        assert user.status().state is UserState.COMPLETED, user.status().reason
        ids = {session(r) for r in endpoint.requests}
        assert ids.isdisjoint(previous_ids)
        assert user.status().run_id not in ids
        previous_ids |= ids
        _assert_adaptation(endpoint.requests, branch)
        _assert_journal(user, chats, endpoint.requests, branch)
        for name in ("events.jsonl", "metadata.json"):
            assert "synthetic-" not in (directory / name).read_text(encoding="utf-8")
