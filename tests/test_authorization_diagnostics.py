"""Authorization snapshots through real FITD and a loopback Chat Completions peer."""

import json
from collections import deque
from dataclasses import FrozenInstanceError
from threading import Event

import pytest
from http_peer import Reply

from tai import (
    ContainsTextEvaluator,
    Message,
    OpenAIChat,
    UserState,
    capture_run,
    save_run,
)
from tai import (
    FITDExperiment as FITDUser,
)
from tai._diagnostics import DiagnosticRedactor
from tai.terminal import run_foreground

TARGET_SECRET = "synthetic-target-secret"
ATTACKER_SECRET = "synthetic-attacker-secret"


def configured_chats(endpoint, target_mode):
    chats, headers = {}, {}
    metadata = {
        TARGET_SECRET: "target entry",
        ATTACKER_SECRET: "attacker entry",
        "[REDACTED]": "literal entry",
        "nested": [{f"key {TARGET_SECRET}": f"value {ATTACKER_SECRET}"}],
    }
    for role, secret, mode in [
        ("Target", TARGET_SECRET, target_mode),
        (
            "Attacker",
            ATTACKER_SECRET,
            "authorization" if target_mode == "api_key" else "api_key",
        ),
    ]:
        credential = secret if mode == "api_key" else f"Custom {secret}"
        headers[role] = f"Bearer {secret}" if mode == "api_key" else credential
        chats[role] = OpenAIChat(
            base_url=endpoint.url + f"/{role.lower()}",
            model="puzzle",
            system=f"{role} {TARGET_SECRET} {ATTACKER_SECRET}",
            generation={"metadata": metadata},
            **{mode: credential},
        )
    return chats, headers, metadata


def assert_safe(serialized):
    assert TARGET_SECRET not in serialized
    assert ATTACKER_SECRET not in serialized


def read_events(directory):
    journal = (directory / "events.jsonl").read_text(encoding="utf-8")
    assert_safe(journal)
    return [json.loads(line) for line in journal.splitlines()]


def assert_settings(settings):
    for role in ("Target", "Attacker"):
        assert settings[role]["generation"]["metadata"] == [
            ["[REDACTED]", "target entry"],
            ["[REDACTED]", "attacker entry"],
            ["[REDACTED]", "literal entry"],
            ["nested", [{"key [REDACTED]": "value [REDACTED]"}]],
        ]


@pytest.mark.parametrize(
    ("target_secret", "attacker_authorization"),
    [
        ("run_started", "timestamp"),
        ("run_id", "kind"),
        ("role", "call_id"),
        ("model", "initial_history"),
        ("synthetic-key", "prefix synthetic-key"),
    ],
)
def test_real_fitd_applies_joint_policy_to_settings_and_entire_serialized_events(
    endpoint, tmp_path, target_secret, attacker_authorization
):
    generation = {
        "metadata": {
            attacker_authorization: "credential entry",
            "[REDACTED]": "literal",
        }
    }
    target = OpenAIChat(
        base_url=endpoint.url + "/target",
        model="puzzle",
        api_key=target_secret,
        generation=generation,
    )
    attacker = OpenAIChat(
        base_url=endpoint.url + "/attacker",
        model="puzzle",
        authorization=attacker_authorization,
    )
    endpoint.scripts = {
        "/v1/attacker/chat/completions": deque(
            Reply("I cannot help with that") for _ in range(3)
        )
    }
    user = FITDUser(
        target,
        attacker,
        Message("Puzzle"),
        ContainsTextEvaluator("CONTROL"),
        journal_dir=tmp_path,
    )
    assert run_foreground(user, artifacts_dir=tmp_path) == 0
    assert user.status().reason == "strategy_exhausted"
    directory = tmp_path / user.status().run_id
    journal = (directory / "events.jsonl").read_text(encoding="utf-8")
    metadata = (directory / "metadata.json").read_text(encoding="utf-8")
    for secret in (target_secret, attacker_authorization):
        assert secret not in journal + metadata
        assert all(secret not in event.dump() for event in user.journal.events())
    records = [json.loads(line) for line in journal.splitlines()]
    assert records == [json.loads(event.dump()) for event in user.journal.events()]
    if target_secret == "run_id":
        # Both envelope keys survive as separate pairs after becoming identical.
        assert [key for key, _ in records[0]] == [
            "[REDACTED]",
            "sequence",
            "[REDACTED]",
            "timestamp",
            "data",
        ]
        assert records[0][0][1] == user.status().run_id
        assert records[0][2][1] == "run_started"
    settings = next(
        e.data["settings"] for e in user.journal.events() if e.kind == "run_configured"
    )
    target_settings = settings["Target"]
    if target_secret == "model":
        assert [pair for pair in target_settings if pair[0] == "[REDACTED]"] == [
            ["[REDACTED]", "puzzle"],
            ["[REDACTED]", []],
        ]
        generation_snapshot = next(
            value for key, value in target_settings if key == "generation"
        )
    else:
        generation_snapshot = target_settings["generation"]
    assert generation_snapshot["metadata"] == [
        ["[REDACTED]", "credential entry"],
        ["[REDACTED]", "literal"],
    ]


@pytest.mark.parametrize("target_mode", ["api_key", "authorization"])
def test_real_fitd_saves_safe_settings_even_when_target_is_never_called(
    endpoint, tmp_path, target_mode
):
    chats, headers, generation_metadata = configured_chats(endpoint, target_mode)
    endpoint.scripts = {
        "/v1/attacker/chat/completions": deque(
            Reply("I cannot help with that") for _ in range(3)
        )
    }

    from tai import EvaluatorSetup

    def settings():
        return {
            "type": "contains_text",
            "criterion": "CONTROL",
            "notes": ({7: f"{TARGET_SECRET} {ATTACKER_SECRET}"},),
            "labels": {
                TARGET_SECRET: "target note",
                ATTACKER_SECRET: "attacker note",
            },
        }

    user = FITDUser(
        chats["Target"],
        chats["Attacker"],
        Message("Explain the fictional puzzle"),
        EvaluatorSetup(ContainsTextEvaluator("CONTROL"), configuration=settings),
        journal_dir=tmp_path,
    )
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.COMPLETED, status.reason
    assert status.reason == "strategy_exhausted"
    assert status.progress.target_calls == 0
    assert status.progress.attacker_calls == 3
    assert user.result() is None and status.evaluation is None
    directory = save_run(capture_run(user, journal=user.journal), tmp_path)
    events = read_events(directory)
    metadata_text = (directory / "metadata.json").read_text(encoding="utf-8")
    assert_safe(metadata_text)
    metadata = json.loads(metadata_text)
    assert_settings(
        next(e["data"]["settings"] for e in events if e["kind"] == "run_configured")
    )
    assert metadata["settings"] == next(
        e["data"]["settings"] for e in events if e["kind"] == "run_configured"
    )
    assert metadata["settings"]["evaluator"]["notes"] == [
        {"7": "[REDACTED] [REDACTED]"}
    ]
    assert metadata["settings"]["evaluator"]["labels"] == [
        ["[REDACTED]", "target note"],
        ["[REDACTED]", "attacker note"],
    ]
    assert metadata["result"] is None
    assert not (directory / "result.txt").exists()
    assert events == [json.loads(event.dump()) for event in user.journal.events()]
    assert len(endpoint.requests) == 3
    for path, request_headers, body in endpoint.requests:
        assert path == "/v1/attacker/chat/completions"
        assert request_headers["Authorization"] == headers["Attacker"]
        assert body["metadata"] == generation_metadata


@pytest.mark.parametrize("target_mode", ["api_key", "authorization"])
@pytest.mark.parametrize("criterion", ["CONTROL", TARGET_SECRET])
def test_real_fitd_redacts_echoes_before_writing_but_evaluates_full_message(
    endpoint, tmp_path, target_mode, criterion
):
    chats, headers, generation_metadata = configured_chats(endpoint, target_mode)
    endpoint.scripts = {
        "/v1/target/chat/completions": deque(
            [Reply("Hint\n", echo_authorization=True) for _ in range(4)]
            + [Reply("CONTROL\n", echo_authorization=True)]
        ),
        "/v1/attacker/chat/completions": deque(
            [Reply("Opening ", echo_authorization=True)]
            + [Reply(text) for text in ["L1", "L2", "L3"] + ["yes"] * 5]
        ),
    }
    entered, release = Event(), Event()
    evaluated = []

    class RecordingEvaluator(ContainsTextEvaluator):
        def evaluate(self, base, result):
            evaluated.append(result)
            entered.set()
            assert release.wait(10), "test did not release the evaluator"
            return super().evaluate(base, result)

    draft = Message(f"Draft {TARGET_SECRET}")
    prepared = draft.transform(str.replace, "Draft", "Prepared")
    base = prepared.spawn(f"Puzzle {ATTACKER_SECRET}").transform(
        lambda text: text + " solution"
    )
    base_dump = base.dump()
    from tai import EvaluatorSetup

    user = FITDUser(
        chats["Target"],
        chats["Attacker"],
        base,
        EvaluatorSetup(
            RecordingEvaluator(criterion),
            configuration=lambda: {"type": "contains_text", "criterion": criterion},
        ),
        journal_dir=tmp_path,
    )
    expected = f"CONTROL\n{headers['Target']}"
    user.start()
    try:
        assert entered.wait(10), user.status()
        assert user.status().state is UserState.RUNNING
        result = user.result()
        assert result.to_string() == expected
        assert evaluated == [result] and evaluated[0] is result
        assert chats["Target"].history()[-1].message is result
        assert json.loads(result.dump())["origin"]["origin"] == json.loads(base_dump)
        result_dump = result.dump()
        with pytest.raises(FrozenInstanceError):
            result.text = "changed"
        directory = tmp_path / user.status().run_id
        # Inspect already flushed data while the run is blocked in the evaluator:
        # final-file cleanup cannot satisfy this check.
        events_before = read_events(directory)
        assert events_before[-1]["kind"] == "evaluation_started"
        assert events_before == [
            json.loads(event.dump()) for event in user.journal.events()
        ]
        publication = next(
            event["data"]["result"]
            for event in events_before
            if event["kind"] == "result_published"
            and event["data"]["result"]["id"] == result.id
        )
        safe_answer = (
            "CONTROL\nBearer [REDACTED]"
            if target_mode == "api_key"
            else "CONTROL\n[REDACTED]"
        )
        assert publication["text"] == safe_answer
        assert (
            publication["origin"]["origin"]["origin"]["text"] == "Prepared [REDACTED]"
        )
        assert publication["origin"]["origin"]["transformations"][0]["text"] == (
            "Puzzle [REDACTED] solution"
        )
        assert not (directory / "result.txt").exists()
    finally:
        release.set()
        assert user.wait(10)
    status = user.status()
    assert status.state is UserState.COMPLETED, status.reason
    assert status.reason == "evaluator_success"
    save_run(capture_run(user, journal=user.journal), tmp_path)
    assert status.evaluation.success is True
    assert status.evaluation.result_id == result.id
    assert user.result() is result and result.dump() == result_dump
    assert base.dump() == base_dump and prepared.revert(1) is draft
    assert base.revert(1).to_string() == f"Puzzle {ATTACKER_SECRET}"
    events = read_events(directory)
    assert events[: len(events_before)] == events_before
    assert events == [json.loads(event.dump()) for event in user.journal.events()]
    metadata_text = (directory / "metadata.json").read_text(encoding="utf-8")
    assert_safe(metadata_text)
    metadata = json.loads(metadata_text)
    assert metadata["result"] == publication
    assert metadata["evaluation"] == {"result_id": result.id, "success": True}
    assert metadata["settings"]["evaluator"]["criterion"] == (
        "[REDACTED]" if criterion == TARGET_SECRET else "CONTROL"
    )
    assert_settings(metadata["settings"])
    assert (directory / "result.txt").read_text(encoding="utf-8") == expected
    target_calls = []
    for path, request_headers, body in endpoint.requests:
        role = "Target" if "/target/" in path else "Attacker"
        assert request_headers["Authorization"] == headers[role]
        assert body["metadata"] == generation_metadata
        assert body["messages"][0]["content"] == (
            f"{role} {TARGET_SECRET} {ATTACKER_SECRET}"
        )
        if role == "Target":
            target_calls.append(body)
    assert len(target_calls) == 5
    assert (
        target_calls[0]["messages"][-1]["content"] == f"Opening {headers['Attacker']}"
    )
    assert target_calls[1]["messages"][2]["content"] == f"Hint\n{headers['Target']}"


def test_json_escaped_redaction_preserves_longest_match_and_colliding_keys():
    secret = 'synthetic"\\credential'
    encoded = json.dumps(secret, ensure_ascii=False)[1:-1]
    redactor = DiagnosticRedactor.combine(
        DiagnosticRedactor(['synthetic"']), DiagnosticRedactor([secret])
    )
    original = {
        secret: "raw key",
        encoded: {"nested": [secret, encoded]},
        "[REDACTED]": "literal marker key",
    }
    snapshot = redactor.snapshot(original)
    assert snapshot == [
        ["[REDACTED]", "raw key"],
        ["[REDACTED]", {"nested": ["[REDACTED]", "[REDACTED]"]}],
        ["[REDACTED]", "literal marker key"],
    ]
    assert original[encoded]["nested"] == [secret, encoded]
    assert redactor.snapshot(snapshot) == snapshot
    assert redactor.snapshot({"token": "unrelated"}, mask_fields=True) == {
        "token": "[REDACTED]"
    }


@pytest.mark.parametrize("marker_secret", ["[REDACTED]", "REDACTED", "D]["])
def test_json_escaped_redaction_never_recreates_credentials_through_markers(
    marker_secret,
):
    secret = 'synthetic"\\credential'
    encoded = json.dumps(secret, ensure_ascii=False)[1:-1]
    redactor = DiagnosticRedactor([secret, marker_secret])
    assert redactor.text(encoded + encoded) == ""
    assert redactor.snapshot(
        {encoded + encoded: "first", secret + secret: "second"}
    ) == [
        ["", "first"],
        ["", "second"],
    ]
