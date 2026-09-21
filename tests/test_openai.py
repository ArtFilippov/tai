"""Public chat configuration, wire history, response and redaction contracts."""

import json
import math
from collections import deque
from importlib.metadata import version
from typing import Protocol, cast

import pytest
from chats_assertions import HttpEndpoint, at, list_at
from http_peer import Reply

import tai
from tai import Message, OpenAIChat


class InvalidOpenAIInput(Protocol):
    def __call__(self, **options: object) -> OpenAIChat: ...


def test_chat_posts_configuration_and_its_complete_immutable_dialogue(endpoint: HttpEndpoint) -> None:
    chat = OpenAIChat(
        base_url=endpoint.url, model="fictional-puzzle", api_key="synthetic-test-key",
        system="Speak as the fictional puzzle keeper.",
        generation={"temperature": 0.25, "max_tokens": 80, "seed": 7},
    )
    first = chat.send(Message("Which key?"))
    snapshot = chat.history()
    second = chat.send(Message("Which door?"))
    assert first.to_string() == "The silver key"
    assert second.to_string() == "The blue door"
    assert len(snapshot) == 3
    assert snapshot[-1].message is first
    assert len(chat.history()) == 5
    path, headers, body = endpoint.requests[1]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer synthetic-test-key"
    assert headers["User-Agent"] == f"tai/{version('tai')}"
    assert "x-opencode-session" not in {key.lower() for key in headers}
    assert body == {
        "model": "fictional-puzzle", "messages": [
            {"role": "system", "content": "Speak as the fictional puzzle keeper."},
            {"role": "user", "content": "Which key?"},
            {"role": "assistant", "content": "The silver key"},
            {"role": "user", "content": "Which door?"},
        ], "temperature": 0.25, "max_tokens": 80, "seed": 7, "n": 1, "stream": False,
    }


@pytest.mark.parametrize(("reply", "error_name"), [
    (Reply(status=401, payload=b"synthetic-secret"), "OpenAIAuthenticationError"),
    (Reply(status=403), "OpenAIAuthenticationError"),
    (Reply(status=429), "OpenAIRateLimitError"), (Reply(status=503), "OpenAIHTTPError"),
    (Reply(status=307), "OpenAIHTTPError"), (Reply(disconnect=True), "OpenAITransportError"),
    (Reply(payload=b"not JSON: synthetic-secret"), "OpenAIJSONError"),
    (Reply(payload=b"\xff"), "OpenAIJSONError"),
    (Reply(payload=b'{"choices": []}'), "OpenAIContentError"),
    (Reply(payload=b'{"choices": [{"message": {"content": null}}]}'), "OpenAIContentError"),
    (Reply(payload=b'{"choices": [{"message": {"content": ["text"]}}]}'), "OpenAIContentError"),
    (Reply(payload=b'{"error": {"message": "synthetic-secret"}}'), "OpenAIContentError"),
])
def test_failed_response_is_typed_safe_and_never_completes_an_exchange(
    endpoint: HttpEndpoint, reply: Reply, error_name: str,
) -> None:
    endpoint.scripts["/v1/chat/completions"] = deque([
        Reply("Completed puzzle hint"), reply, Reply("Recovered hint"),
    ])
    chat = OpenAIChat(base_url=endpoint.url, model="puzzle", api_key="synthetic-secret")
    chat.send(Message("First hint"))
    history = chat.history()
    with pytest.raises(getattr(tai, error_name)) as caught:
        chat.send(Message("Next hint"))
    assert "synthetic-secret" not in str(caught.value)
    assert "synthetic-secret" not in repr(caught.value)
    assert chat.history() == history
    if not reply.disconnect:
        assert caught.value.status_code == reply.status
    assert chat.send(Message("Retry hint")).to_string() == "Recovered hint"
    assert "Next hint" not in json.dumps(endpoint.requests[-1][2]["messages"])


@pytest.mark.parametrize("changes", [
    {"api_key": "key", "authorization": "Bearer other"},
    {"authorization": "Bearer key\r\nInjected: bad"},
    {"base_url": "http://name:password@localhost/v1"},
    {"base_url": "http://localhost/v1?api_key=secret"}, {"base_url": "file:///tmp/provider"},
    {"timeout": 0}, {"opencode_session": "true"}, {"opencode_session": 1},
    {"timeout": math.inf}, {"generation": {"messages": []}},
    {"generation": {"model": "other"}}, {"generation": {"stream": True}},
    {"generation": {"n": 2}}, {"generation": {"temperature": math.nan}},
])
def test_invalid_connection_configuration_fails_before_io(changes: dict[str, object]) -> None:
    construct = cast(InvalidOpenAIInput, OpenAIChat)
    with pytest.raises(ValueError):
        construct(**{"base_url": "http://127.0.0.1/v1", "model": "puzzle", **changes})


def test_generation_and_settings_are_detached_from_caller_mutations(endpoint: HttpEndpoint) -> None:
    generation = {"stop": ["END"]}
    chat = OpenAIChat(base_url=endpoint.url, model="puzzle", generation=generation)
    generation["stop"].append("CHANGED")
    settings = chat.settings()
    list_at(settings, "generation", "stop").append("MUTATED")
    chat.send(Message("First clue"))
    assert at(endpoint.requests[0][2], "stop") == ["END"]
    assert "Authorization" not in endpoint.requests[0][1]
    assert at(chat.settings(), "authorization") == "none"


def test_settings_redact_credentials_even_when_repeated_in_generation_metadata(
    endpoint: HttpEndpoint,
) -> None:
    chat = OpenAIChat(
        base_url=endpoint.url, model="puzzle", api_key="synthetic-secret",
        generation={"metadata": {"label": "synthetic-secret", "api_key": "another-secret"}},
    )
    settings = chat.settings()
    assert "synthetic-secret" not in json.dumps(settings)
    assert "another-secret" not in json.dumps(settings)
    assert at(settings, "generation", "metadata") == {
        "label": "[REDACTED]", "api_key": "[REDACTED]",
    }
    chat.send(Message("Puzzle clue"))
    assert at(endpoint.requests[0][2], "metadata", "label") == "synthetic-secret"


def test_settings_redact_credentials_in_dictionary_keys() -> None:
    secret = "synthetic-review-key"
    chat = OpenAIChat(
        base_url="http://127.0.0.1/v1", model="puzzle", api_key=secret,
        generation={"metadata": {"label": secret, secret: "run-label"}},
    )
    settings = chat.settings()
    assert at(settings, "generation", "metadata", "label") == "[REDACTED]"
    assert secret not in json.dumps(settings)
    assert at(settings, "generation", "metadata", "[REDACTED]") == "run-label"


@pytest.mark.parametrize("secret", ["[REDACTED]", "REDACTED", "D]["])
def test_settings_never_reintroduce_a_credential_through_redaction_markers(secret: str) -> None:
    chat = OpenAIChat(
        base_url="http://127.0.0.1/v1", model="puzzle", api_key=secret,
        generation={"metadata": {
            "repeat": secret + secret, "token": "other synthetic value", "keep": "unrelated data",
        }},
    )
    settings = chat.settings()
    assert secret not in json.dumps(settings)
    assert not at(settings, "generation", "metadata", "repeat")
    assert at(settings, "generation", "metadata", "keep") == "unrelated data"
    assert chat.settings() == settings


def _assert_collision_settings(chat: OpenAIChat, secret: str) -> None:
    settings = chat.settings()
    assert at(settings, "generation", "metadata") == [
        ["[REDACTED]", "literal marker"], ["[REDACTED]", "credential key"],
        ["prefix [REDACTED]", {"nested": ["value [REDACTED]", {"[REDACTED]": "leaf"}]}],
        ["prefix [REDACTED]", "literal prefix"],
        ["keep", {"label": "unrelated data", "api_key": "[REDACTED]"}],
    ]
    assert secret not in json.dumps(settings)
    assert chat.settings() == settings


@pytest.mark.parametrize("mode", ["api_key", "authorization"])
def test_settings_preserve_colliding_entries_and_original_http_exchange(
    endpoint: HttpEndpoint, mode: str,
) -> None:
    secret = "synthetic-review-key"
    credential = secret if mode == "api_key" else f"Custom {secret}"
    header = f"Bearer {secret}" if mode == "api_key" else credential
    metadata = {
        "[REDACTED]": "literal marker", secret: "credential key",
        f"prefix {secret}": {"nested": [f"value {credential}", {secret: "leaf"}]},
        "prefix [REDACTED]": "literal prefix",
        "keep": {"label": "unrelated data", "api_key": "other synthetic value"},
    }
    endpoint.scripts["/v1/chat/completions"] = deque([
        Reply("CONTROL ", echo_authorization=True), Reply("Next answer"),
    ])
    chat = OpenAIChat(
        base_url=endpoint.url, model=f"puzzle {secret}", system=f"Instructions {credential}",
        generation={"metadata": metadata}, api_key=credential if mode == "api_key" else None,
        authorization=credential if mode == "authorization" else None,
    )
    _assert_collision_settings(chat, secret)
    base = Message(f"Draft {credential}")
    prepared = base.transform(str.replace, "Draft", "Prepared")
    question = prepared.spawn(f"Question {secret}").transform(str.upper)
    before = question.dump()
    answer = chat.send(question)
    assert answer.to_string() == f"CONTROL {header}"
    assert chat.history()[-1].message is answer
    assert json.loads(answer.dump())["origin"] == json.loads(before)
    assert question.revert(1).base() == f"Question {secret}"
    assert prepared.revert(1) is base
    history = chat.history()
    answer_dump = answer.dump()
    assert secret not in json.dumps(chat.settings())
    chat.send(Message("Next clue"))
    assert question.dump() == before and answer.dump() == answer_dump
    assert chat.history()[:-2] == history
    for _, headers, body in endpoint.requests:
        assert headers["Authorization"] == header
        assert body["model"] == f"puzzle {secret}"
        assert at(body, "metadata") == metadata
        assert body["messages"][0]["content"] == f"Instructions {credential}"
    assert endpoint.requests[-1][2]["messages"][-2]["content"] == f"CONTROL {header}"


def test_external_providers_are_rejected_by_the_offline_guard_in_http_workers() -> None:
    chat = OpenAIChat(base_url="http://provider.invalid/v1", model="puzzle", timeout=2)
    with pytest.raises(tai.OpenAITransportError):
        chat.send(Message("Fictional clue"))
    assert chat.history() == ()


def test_truncated_body_preserves_observed_bytes_as_safe_diagnostics(endpoint: HttpEndpoint) -> None:
    payload = "OBSERVED_PARTIAL π synthetic-secret".encode()
    endpoint.scripts["/v1/chat/completions"] = deque([
        Reply("Saved hint"), Reply(payload=payload, truncated=True), Reply("Recovered"),
    ])
    chat = OpenAIChat(base_url=endpoint.url, model="puzzle", api_key="synthetic-secret")
    chat.send(Message("First clue"))
    history = chat.history()
    with pytest.raises(tai.OpenAITransportError) as caught:
        chat.send(Message("Truncated clue"))
    error = caught.value
    assert error.response_bytes == len(payload)
    assert error.status_code == 200
    assert error.diagnostic == {
        "status_code": 200, "response_bytes": len(payload),
        "response_fragment": "OBSERVED_PARTIAL π [REDACTED]",
    }
    assert "synthetic-secret" not in str(error) + json.dumps(error.diagnostic)
    assert "IncompleteRead" in str(error)
    assert chat.history() == history
    assert chat.send(Message("Retry clue")).to_string() == "Recovered"
    assert "Truncated clue" not in json.dumps(endpoint.requests[-1][2]["messages"])


def test_http_checkpoint_restore_and_reset_control_the_next_actual_request(
    endpoint: HttpEndpoint,
) -> None:
    endpoint.scripts["/v1/chat/completions"] = deque(
        Reply(text) for text in ("First", "Discarded", "Replacement", "Fresh")
    )
    chat = OpenAIChat(base_url=endpoint.url, model="puzzle", system="Puzzle rules")
    chat.send(Message("Opening"))
    point = chat.checkpoint()
    retained = chat.history()
    chat.send(Message("Discard this clue"))
    chat.restore(point)
    assert chat.history() == retained
    chat.send(Message("Alternative clue"))
    assert endpoint.requests[2][2]["messages"] == [
        {"role": "system", "content": "Puzzle rules"}, {"role": "user", "content": "Opening"},
        {"role": "assistant", "content": "First"}, {"role": "user", "content": "Alternative clue"},
    ]
    chat.reset()
    with pytest.raises(ValueError, match="checkpoint"):
        chat.restore(point)
    chat.send(Message("New conversation"))
    assert endpoint.requests[3][2]["messages"] == [
        {"role": "system", "content": "Puzzle rules"},
        {"role": "user", "content": "New conversation"},
    ]
