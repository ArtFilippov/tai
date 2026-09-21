"""Live and saved evidence checks for evaluator credential redaction."""

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from app_test_support import HttpPeer
from http_peer import Reply
from test_cli import read_artifacts

from alert.values import object_value, text_value
from app.configuration import RoleProfile, load_profile
from tai import FITDExperiment, Message, UserState, capture_run, save_run


@dataclass
class RedactionCase:
    root: Path
    profile: RoleProfile
    user: FITDExperiment
    base: Message
    final: str
    secrets: list[str]
    expected: dict[str, str]


@dataclass
class LiveEvidence:
    result: Message
    events: list[object]
    request: dict[str, object]
    raw_request: str


def assert_safe_request(snapshot: object, expected: dict[str, str]) -> None:
    # Inspect the decoded payload, not just literal matches in outer JSON.
    # Message snapshots repeat the original request in both text and base.
    document = object_value(snapshot, "message snapshot")
    for field in ("base", "text"):
        text = text_value(document[field], field)
        assert json.loads(text.split("\n\nINPUT_JSON:\n", 1)[1]) == expected


def assert_safe_tree(value: object, secrets: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in cast(dict[str, object], value).items():
            assert_safe_tree(key, secrets)
            assert_safe_tree(item, secrets)
    elif isinstance(value, list):
        for item in cast(list[object], value):
            assert_safe_tree(item, secrets)
    elif isinstance(value, str):
        _assert_safe_text(value, secrets)


def _assert_safe_text(text: str, secrets: list[str]) -> None:
    for secret in secrets:
        assert secret not in text
        assert json.dumps(secret, ensure_ascii=False)[1:-1] not in text


def make_case(
    endpoint: HttpPeer, config: Path, monkeypatch: pytest.MonkeyPatch, escaped: str, verdict: str,
) -> tuple[RedactionCase, Reply]:
    secrets = [f"synthetic{escaped}{role}-secret" for role in ("target", "attacker", "judge")]
    target_secret, attacker_secret, judge_secret = secrets
    monkeypatch.setenv("TAI_TEST_TARGET_KEY", target_secret)
    monkeypatch.setenv("TAI_TEST_ATTACKER_AUTH", f"Custom {attacker_secret}")
    monkeypatch.setenv("TAI_TEST_EVALUATOR_KEY", judge_secret)
    base = Message(f"Explicit base {target_secret}; Custom {attacker_secret}; {judge_secret}")
    final = f"Presented {judge_secret}; {target_secret}; {attacker_secret}\nFull answer π\n"
    endpoint.scripts["/v1/target/chat/completions"][-1] = Reply(final)
    held = Reply(verdict, hold=True)
    endpoint.scripts["/v1/evaluator/chat/completions"] = deque([held])
    profile = load_profile(config)
    root = config.parent / "runs"
    user = profile.create_experiment(base, root)
    expected = {"base_request": "Explicit base [REDACTED]; [REDACTED]; [REDACTED]",
                "result": "Presented [REDACTED]; [REDACTED]; [REDACTED]\nFull answer π\n"}
    return RedactionCase(root, profile, user, base, final, secrets, expected), held


def capture_live(case: RedactionCase, endpoint: HttpPeer, held: Reply) -> LiveEvidence:
    assert held.entered.wait(10), case.user.status()
    assert case.user.status().state is UserState.RUNNING
    run_id = case.user.status().run_id
    assert run_id is not None
    directory = case.root / run_id
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert events == [json.loads(event.dump()) for event in case.user.journal.events()]
    assert events[-1]["kind"] == "call_started"
    assert events[-1]["data"]["role"] == "Evaluator"
    request = object_value(events[-1]["data"]["input"], "request")
    assert_safe_request(request, case.expected)
    assert_safe_tree(events, case.secrets)
    for event in case.user.journal.events():
        assert_safe_tree(event.data, case.secrets)
    result = case.user.result()
    assert result is not None and result.to_string() == case.final
    assert not (directory / "result.txt").exists()
    wire = endpoint.requests[-1]
    assert wire[1]["Authorization"] == f"Bearer {case.secrets[2]}"
    raw_request = wire[2]["messages"][-1]["content"]
    assert json.loads(raw_request.split("\n\nINPUT_JSON:\n", 1)[1]) == {
        "base_request": case.base.to_string(), "result": case.final,
    }
    return LiveEvidence(result, events, request, raw_request)


def assert_saved(case: RedactionCase, evidence: LiveEvidence, verdict: str, base_dump: str) -> None:
    valid = verdict == '{"success": true}'
    assert case.user.status().state is (UserState.COMPLETED if valid else UserState.FAILED)
    save_run(capture_run(case.user, journal=case.user.journal), case.root)
    directory, metadata, events = read_artifacts(case.root)
    assert metadata["evaluation"] == ({"result_id": evidence.result.id, "success": True} if valid else None)
    assert events[:len(evidence.events)] == evidence.events
    assert events == [json.loads(event.dump()) for event in case.user.journal.events()]
    assert_safe_tree(events, case.secrets)
    assert_safe_tree(metadata, case.secrets)
    assert metadata["settings"]["base"]["text"] == case.expected["base_request"]
    completed = [event["data"] for event in events
                 if event["kind"] == "call_completed" and event["data"]["role"] == "Evaluator"]
    assert len(completed) == 1
    assert completed[0]["response"]["text"] == verdict
    assert completed[0]["response"]["origin"]["id"] == evidence.request["id"]
    assert_safe_request(completed[0]["response"]["origin"], case.expected)
    history = metadata["final_settings"]["Evaluator"]["initial_history"]
    assert_safe_request(history[-2]["message"], case.expected)
    assert_safe_request(history[-1]["message"]["origin"], case.expected)
    assert (directory / "result.txt").read_text() == case.final
    assert case.user.result() is evidence.result and evidence.result.to_string() == case.final
    assert case.base.dump() == base_dump
    # Redaction never mutates the actual evaluator conversation or its origins.
    assert case.profile.evaluator.chat is not None
    raw_history = case.profile.evaluator.chat.history()
    assert raw_history[-2].message.to_string() == evidence.raw_request
    assert json.loads(raw_history[-1].message.dump())["origin"]["text"] == evidence.raw_request
