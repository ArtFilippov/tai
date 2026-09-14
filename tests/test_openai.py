"""Public chat and real FITD contracts against a controlled local HTTP peer."""

import json
import os
import select
import signal
import socket
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread, current_thread

import pytest

import tai
from tai import Message, OpenAIChat


@dataclass
class Reply:
    text: str = "Puzzle answer"
    status: int = 200
    payload: bytes | None = None
    hold: bool = False
    partial: bool = False
    partial_bytes: int = 5
    disconnect: bool = False
    truncated: bool = False
    entered: Event = field(default_factory=Event)
    release: Event = field(default_factory=Event)
    disconnected: Event = field(default_factory=Event)


class Endpoint:
    def __init__(self):
        self.requests = []
        self.scripts = {
            "/v1/chat/completions": deque(
                [Reply("The silver key"), Reply("The blue door")]
            )
        }
        self.url = ""


@pytest.fixture
def endpoint():
    peer = Endpoint()
    active = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            peer.requests.append((self.path, dict(self.headers), body))
            reply = peer.scripts[self.path].popleft()
            active.append(reply)
            if reply.disconnect:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            payload = reply.payload
            if payload is None:
                payload = json.dumps(
                    {
                        "choices": [
                            {"message": {"role": "assistant", "content": reply.text}}
                        ]
                    }
                ).encode()

            def headers():
                self.send_response(reply.status)
                self.send_header("Content-Type", "application/json")
                self.send_header(
                    "Content-Length",
                    str(len(payload) + (100 if reply.truncated else 0)),
                )
                self.send_header("Location", peer.url + "/redirected")
                self.end_headers()

            try:
                if reply.partial:
                    headers()
                    self.wfile.write(payload[: reply.partial_bytes])
                    self.wfile.flush()
                reply.entered.set()
                while reply.hold and not reply.release.is_set():
                    readable, _, _ = select.select([self.connection], [], [], 0.02)
                    if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                        reply.disconnected.set()
                        return
                if reply.partial:
                    self.wfile.write(payload[reply.partial_bytes :])
                else:
                    headers()
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                reply.disconnected.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    peer.url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield peer
    finally:
        for reply in active:
            reply.release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def test_chat_posts_configuration_and_its_complete_immutable_dialogue(endpoint):
    chat = OpenAIChat(
        base_url=endpoint.url,
        model="fictional-puzzle",
        api_key="synthetic-test-key",
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
    assert body == {
        "model": "fictional-puzzle",
        "messages": [
            {"role": "system", "content": "Speak as the fictional puzzle keeper."},
            {"role": "user", "content": "Which key?"},
            {"role": "assistant", "content": "The silver key"},
            {"role": "user", "content": "Which door?"},
        ],
        "temperature": 0.25,
        "max_tokens": 80,
        "seed": 7,
        "n": 1,
        "stream": False,
    }


def gate_worker_input(monkeypatch):
    """Stop a real child before stdin reads, with a known 4096-byte pipe capacity."""
    workers, communicators = [], []
    filled = Event()
    written = {}
    popen, write = subprocess.Popen, os.write

    class GatedWorker(popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, "pipesize": 4096})
            self.send_signal(signal.SIGSTOP)
            written[self.stdin.fileno()] = 0
            workers.append(self)

        def communicate(self, *args, **kwargs):
            communicators.append(current_thread())
            return super().communicate(*args, **kwargs)

    def observe_write(fd, data):
        count = write(fd, data)
        if fd in written:
            written[fd] += count
            if written[fd] >= 4096:
                filled.set()
        return count

    monkeypatch.setattr(subprocess, "Popen", GatedWorker)
    monkeypatch.setattr(os, "write", observe_write)
    return filled, workers, communicators


@pytest.mark.parametrize("large_source", ["initial", "assistant_history"])
def test_large_request_reaches_endpoint_after_stdin_backpressure(
    endpoint, monkeypatch, large_source
):
    initial = "Question:\n" + "q" * 150000 + "\nEND π"
    assistant = "Answer:\n" + "a" * 150054 + "\nEND π"
    chat = OpenAIChat(
        base_url=endpoint.url, model="puzzle", system="Puzzle rules", timeout=3
    )
    if large_source == "assistant_history":
        endpoint.scripts["/v1/chat/completions"] = deque(
            [Reply(assistant), Reply("Done")]
        )
        assert chat.send(Message("Opening")).to_string() == assistant
        message = Message("Next clue")
        expected = [
            {"role": "system", "content": "Puzzle rules"},
            {"role": "user", "content": "Opening"},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": "Next clue"},
        ]
    else:
        endpoint.scripts["/v1/chat/completions"] = deque([Reply("Done")])
        message = Message(initial)
        expected = [
            {"role": "system", "content": "Puzzle rules"},
            {"role": "user", "content": initial},
        ]
    previous = chat.history()
    calls = len(endpoint.requests)
    filled, workers, communicators = gate_worker_input(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(chat.send, message)
        try:
            assert filled.wait(2), (
                "large request did not fill the stopped worker's pipe"
            )
            # Explicit backpressure, not a startup delay: the child cannot read.
            # The blocked send must remain pending across control polling cycles.
            with pytest.raises(TimeoutError):
                pending.result(timeout=0.1)
            assert chat.history() == previous
            assert len(endpoint.requests) == calls
            workers[0].send_signal(signal.SIGCONT)
            assert pending.result(timeout=5).to_string() == "Done"
            assert endpoint.requests[-1][2]["messages"] == expected
            assert len(endpoint.requests) == calls + 1
            assert chat.history()[:-2] == previous
            assert chat.history()[-2].message is message
        finally:
            chat.cancel()
            for worker in workers:
                if worker.poll() is None:
                    worker.send_signal(signal.SIGCONT)
    for worker in workers:
        assert worker.returncode is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(worker.pid, os.WNOHANG)
    assert all(not thread.is_alive() for thread in communicators)


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_large_blocked_input_can_be_interrupted_and_all_local_io_is_joined(
    endpoint, monkeypatch, stop
):
    endpoint.scripts["/v1/chat/completions"] = deque(
        [Reply("Saved"), Reply("Recovered")]
    )
    chat = OpenAIChat(
        base_url=endpoint.url, model="puzzle", timeout=0.5 if stop == "timeout" else 120
    )
    chat.send(Message("Opening"))
    previous = chat.history()
    with monkeypatch.context() as patch:
        filled, workers, communicators = gate_worker_input(patch)
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = executor.submit(chat.send, Message("x" * 150000))
            try:
                assert filled.wait(2)
                assert not pending.done()
                assert len(endpoint.requests) == 1
                if stop == "cancel":
                    assert executor.submit(chat.cancel).result(timeout=1) is None
                    expected = tai.ChatCancelled
                else:
                    expected = tai.OpenAITimeoutError
                with pytest.raises(expected) as caught:
                    pending.result(timeout=2)
                assert caught.value.diagnostic is None
                assert chat.history() == previous
                assert len(endpoint.requests) == 1
                # No SIGCONT: kill must end the stopped process even while the
                # communicator is unable to finish writing the full request.
                assert len(communicators) == 1
                assert all(not thread.is_alive() for thread in communicators)
                for worker in workers:
                    assert worker.returncode is not None
                    assert (
                        worker.stdin.closed
                        and worker.stdout.closed
                        and worker.stderr.closed
                    )
                    with pytest.raises(ChildProcessError):
                        os.waitpid(worker.pid, os.WNOHANG)
            finally:
                chat.cancel()
                for worker in workers:
                    if worker.poll() is None:
                        worker.send_signal(signal.SIGCONT)
    assert chat.send(Message("Retry")).to_string() == "Recovered"
    assert endpoint.requests[-1][2]["messages"] == [
        {"role": "user", "content": "Opening"},
        {"role": "assistant", "content": "Saved"},
        {"role": "user", "content": "Retry"},
    ]


def test_communication_error_joins_io_thread_and_reaps_worker(endpoint, monkeypatch):
    workers, communicators = [], []
    popen = subprocess.Popen

    class FailedCommunication(popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            workers.append(self)

        def communicate(self, *args, **kwargs):
            communicators.append(current_thread())
            raise OSError("synthetic pipe failure")

    monkeypatch.setattr(subprocess, "Popen", FailedCommunication)
    chat = OpenAIChat(base_url=endpoint.url, model="puzzle")
    with pytest.raises(tai.OpenAITransportError, match="worker communication failed"):
        chat.send(Message("Clue"))
    assert endpoint.requests == []
    assert chat.history() == ()
    assert len(communicators) == 1
    assert all(not thread.is_alive() for thread in communicators)
    for worker in workers:
        assert worker.returncode is not None
        assert worker.stdin.closed and worker.stdout.closed and worker.stderr.closed
        with pytest.raises(ChildProcessError):
            os.waitpid(worker.pid, os.WNOHANG)


@pytest.mark.parametrize(
    ("reply", "error_name"),
    [
        (Reply(status=401, payload=b"synthetic-secret"), "OpenAIAuthenticationError"),
        (Reply(status=403), "OpenAIAuthenticationError"),
        (Reply(status=429), "OpenAIRateLimitError"),
        (Reply(status=503), "OpenAIHTTPError"),
        (Reply(status=307), "OpenAIHTTPError"),
        (Reply(disconnect=True), "OpenAITransportError"),
        (Reply(payload=b"not JSON: synthetic-secret"), "OpenAIJSONError"),
        (Reply(payload=b"\xff"), "OpenAIJSONError"),
        (Reply(payload=b'{"choices": []}'), "OpenAIContentError"),
        (
            Reply(payload=b'{"choices": [{"message": {"content": null}}]}'),
            "OpenAIContentError",
        ),
        (
            Reply(payload=b'{"choices": [{"message": {"content": ["text"]}}]}'),
            "OpenAIContentError",
        ),
        (
            Reply(payload=b'{"error": {"message": "synthetic-secret"}}'),
            "OpenAIContentError",
        ),
    ],
)
def test_failed_response_is_typed_safe_and_never_completes_an_exchange(
    endpoint, reply, error_name
):
    endpoint.scripts["/v1/chat/completions"] = deque(
        [Reply("Completed puzzle hint"), reply, Reply("Recovered hint")]
    )
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


@pytest.mark.parametrize("partial", [False, True], ids=["headers", "body"])
@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_interruption_closes_local_io_before_the_server_releases_a_response(
    endpoint, partial, stop, monkeypatch
):
    workers = []
    popen = subprocess.Popen

    def observe_worker(*args, **kwargs):
        worker = popen(*args, **kwargs)
        workers.append(worker)
        return worker

    # Observe the real OS boundary; do not replace transport or strategy behavior.
    monkeypatch.setattr(subprocess, "Popen", observe_worker)
    held = Reply(hold=True, partial=partial)
    endpoint.scripts["/v1/chat/completions"] = deque(
        [Reply("Saved hint"), held, Reply("")]
    )
    chat = OpenAIChat(
        base_url=endpoint.url, model="puzzle", timeout=0.5 if stop == "timeout" else 120
    )
    chat.cancel()  # Idle cancellation must not poison the next exchange.
    chat.send(Message("First clue"))
    history = chat.history()
    point = chat.checkpoint()
    with ThreadPoolExecutor(max_workers=1) as worker:
        pending = worker.submit(chat.send, Message("Held clue"))
        try:
            assert held.entered.wait(5)
            for mutate in (chat.reset, lambda: chat.restore(point)):
                with pytest.raises(RuntimeError, match="active exchange"):
                    mutate()
            if stop == "cancel":
                chat.cancel()
                expected = tai.ChatCancelled
            else:
                expected = tai.OpenAITimeoutError
            with pytest.raises(expected) as caught:
                pending.result(timeout=2)
            if not partial:
                assert caught.value.diagnostic is None
            assert not held.release.is_set()
            assert held.disconnected.wait(1), "local socket leaked after send unwound"
            assert chat.history() == history
            assert len(workers) == 2
            for process in workers:
                assert process.returncode is not None
                with pytest.raises(ChildProcessError):
                    os.waitpid(process.pid, os.WNOHANG)
            chat.restore(point)
            assert chat.history() == history
        finally:
            held.release.set()
    chat.cancel()
    assert chat.send(Message("Fresh clue")).to_string() == ""
    assert len(endpoint.requests) == 3
    assert "Held clue" not in json.dumps(endpoint.requests[-1][2])


def puzzle_user(
    endpoint,
    tmp_path,
    *,
    target_replies=None,
    attacker_replies=None,
    target_timeout=120.0,
):
    endpoint.scripts = {
        "/v1/target/chat/completions": deque(
            target_replies
            or [
                Reply(text)
                for text in [
                    "Unrelated story",
                    "Puzzle introduction",
                    "Silver key",
                    "Blue door",
                    "Turn the key",
                    "KEY-FOUND",
                ]
            ]
        ),
        "/v1/attacker/chat/completions": deque(
            attacker_replies
            or [
                Reply(text)
                for text in [
                    "Describe a fictional puzzle",
                    "Puzzle level one",
                    "Puzzle level two",
                    "Puzzle level three",
                    "no",
                    "yes",
                    "yes",
                    "yes",
                    "yes",
                    "yes",
                ]
            ]
        ),
    }
    chats = {
        "Target": OpenAIChat(
            base_url=endpoint.url + "/target",
            model="same-puzzle-model",
            api_key="synthetic-target-secret",
            system="Target puzzle keeper",
            generation={"temperature": 0.25},
            timeout=target_timeout,
        ),
        "Attacker": OpenAIChat(
            base_url=endpoint.url + "/attacker",
            model="same-puzzle-model",
            authorization="Custom synthetic-attacker-secret",
            system="Attacker puzzle author",
            generation={"temperature": 0.75},
        ),
    }
    user = tai.FITDUser(
        chats["Target"],
        chats["Attacker"],
        Message("Explain the final fictional puzzle solution"),
        tai.ContainsTextEvaluator("KEY-FOUND"),
        artifacts_dir=tmp_path,
    )
    return user, chats


def artifacts(user, root):
    directory = root / user.status().run_id
    events = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
    ]
    metadata = json.loads((directory / "metadata.json").read_text())
    assert events == [json.loads(event.dump()) for event in user.events()]
    assert metadata["state"] == user.status().state.value
    serialized = json.dumps([events, metadata])
    assert "synthetic-target-secret" not in serialized
    assert "synthetic-attacker-secret" not in serialized
    return events, metadata


def test_real_fitd_realigns_and_uses_role_specific_http_chats_and_safe_settings(
    endpoint, tmp_path
):
    user, chats = puzzle_user(endpoint, tmp_path)
    user.start()
    assert user.wait(10)
    assert user.status().state is tai.UserState.COMPLETED, user.status().reason
    result = user.result()
    assert result is not None and "KEY-FOUND" in result.to_string()
    assert user.status().evaluation.success is True
    events, metadata = artifacts(user, tmp_path)
    for role in ("Target", "Attacker"):
        calls = [
            request
            for request in endpoint.requests
            if f"/{role.lower()}/" in request[0]
        ]
        assert len(calls) > 4
        instructions = (
            "Target puzzle keeper" if role == "Target" else "Attacker puzzle author"
        )
        assert all(
            request[2]["messages"][0]
            == {
                "role": "system",
                "content": instructions,
            }
            for request in calls
        )
        assert all(request[2]["model"] == "same-puzzle-model" for request in calls)
        if role == "Target":
            for earlier, later in zip(calls, calls[1:], strict=False):
                assert (
                    later[2]["messages"][: len(earlier[2]["messages"])]
                    == earlier[2]["messages"]
                )
            assert len(chats[role].history()) == 1 + 2 * len(calls)
        else:
            assert all(len(request[2]["messages"]) == 2 for request in calls)
            assert len(chats[role].history()) == 3
        settings = metadata["settings"][role]
        assert settings["base_url"] == endpoint.url + f"/{role.lower()}"
        assert settings["timeout"] == 120
        assert settings["generation"] == {
            "temperature": 0.25 if role == "Target" else 0.75,
            "n": 1,
            "stream": False,
        }
        assert settings["authorization"] == (
            "api_key" if role == "Target" else "explicit"
        )
        assert settings["initial_history"] == [
            {
                "role": "system",
                "message": json.loads(chats[role].history()[0].message.dump()),
            }
        ]
    target_calls = [
        request for request in endpoint.requests if "/target/" in request[0]
    ]
    attacker_calls = [
        request for request in endpoint.requests if "/attacker/" in request[0]
    ]
    assert all(
        request[1]["Authorization"] == "Bearer synthetic-target-secret"
        for request in target_calls
    )
    assert all(
        request[1]["Authorization"] == "Custom synthetic-attacker-secret"
        for request in attacker_calls
    )
    assert "Unrelated story" in target_calls[1][2]["messages"][-1]["content"]
    assert (
        "Explain the final fictional puzzle solution"
        in attacker_calls[0][2]["messages"][-1]["content"]
    )
    assert len([event for event in events if event["kind"] == "call_completed"]) == len(
        endpoint.requests
    )
    assert metadata["settings"]["generations"] == 1
    assert metadata["settings"]["max_calls_per_conv"] == 10


@pytest.mark.parametrize(
    "changes",
    [
        {"api_key": "key", "authorization": "Bearer other"},
        {"authorization": "Bearer key\r\nInjected: bad"},
        {"base_url": "http://name:password@localhost/v1"},
        {"base_url": "http://localhost/v1?api_key=secret"},
        {"base_url": "file:///tmp/provider"},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"generation": {"messages": []}},
        {"generation": {"model": "other"}},
        {"generation": {"stream": True}},
        {"generation": {"n": 2}},
        {"generation": {"temperature": float("nan")}},
    ],
)
def test_invalid_connection_configuration_fails_before_io(changes):
    with pytest.raises(ValueError):
        OpenAIChat(**{"base_url": "http://127.0.0.1/v1", "model": "puzzle", **changes})


def test_generation_and_settings_are_detached_from_caller_mutations(endpoint):
    generation = {"stop": ["END"]}
    chat = OpenAIChat(base_url=endpoint.url, model="puzzle", generation=generation)
    generation["stop"].append("CHANGED")
    settings = chat.settings()
    settings["generation"]["stop"].append("MUTATED")
    chat.send(Message("First clue"))
    assert endpoint.requests[0][2]["stop"] == ["END"]
    assert "Authorization" not in endpoint.requests[0][1]
    assert chat.settings()["authorization"] == "none"


@pytest.mark.parametrize(
    ("role", "index"), [("Target", 1), ("Attacker", 0), ("Attacker", 4)]
)
def test_user_stop_cancels_target_initial_and_internal_attacker_http_before_response(
    endpoint, tmp_path, role, index
):
    user, chats = puzzle_user(endpoint, tmp_path)
    held = Reply(hold=True)
    endpoint.scripts[f"/v1/{role.lower()}/chat/completions"][index] = held
    with ThreadPoolExecutor(max_workers=1) as control:
        try:
            assert control.submit(user.start).result(timeout=1) is None
            assert held.entered.wait(5)
            assert (
                control.submit(user.status).result(timeout=1).state
                is tai.UserState.RUNNING
            )
            presented = control.submit(user.result).result(timeout=1)
            histories = {name: chat.history() for name, chat in chats.items()}
            calls = len(endpoint.requests)
            assert control.submit(user.stop).result(timeout=1) is None
            assert user.wait(2), (
                "worker waited for the server instead of cancelling HTTP"
            )
            assert not held.release.is_set()
            assert held.disconnected.wait(1), "local HTTP I/O survived STOPPED"
            assert user.status().state is tai.UserState.STOPPED, user.status().reason
            assert user.result() is presented
            assert len(endpoint.requests) == calls
            assert {name: chat.history() for name, chat in chats.items()} == histories
            events, metadata = artifacts(user, tmp_path)
            assert metadata["reason"] == "user_stop"
            cancelled = [
                event["data"] for event in events if event["kind"] == "call_cancelled"
            ]
            assert len(cancelled) == 1 and cancelled[0]["role"] == role
            assert cancelled[0]["response"] is None
            assert (
                len([event for event in events if event["kind"] == "call_completed"])
                == calls - 1
            )
            assert (
                len([event for event in events if event["kind"] == "call_started"])
                == calls
            )
        finally:
            held.release.set()
            user.stop()
            assert user.wait(5)


@pytest.mark.parametrize(
    ("reply", "error_name"),
    [
        (
            Reply(status=401, payload=b"synthetic-target-secret"),
            "OpenAIAuthenticationError",
        ),
        (Reply(status=429, payload=b"synthetic-target-secret"), "OpenAIRateLimitError"),
        (Reply(status=500), "OpenAIHTTPError"),
        (Reply(disconnect=True), "OpenAITransportError"),
        (Reply(payload=b"not JSON synthetic-target-secret"), "OpenAIJSONError"),
        (Reply(payload=b'{"choices": []}'), "OpenAIContentError"),
        (Reply(hold=True), "OpenAITimeoutError"),
    ],
)
def test_http_failure_marks_real_fitd_failed_and_retains_completed_data(
    endpoint, tmp_path, reply, error_name
):
    user, chats = puzzle_user(endpoint, tmp_path)
    # Fail after the correction was accepted, preserving a real presented result
    # as well as the earlier rejected response in the diagnostic context.
    endpoint.scripts["/v1/target/chat/completions"][2] = reply
    if reply.hold:
        chats["Target"] = OpenAIChat(
            base_url=endpoint.url + "/target",
            model="puzzle",
            api_key="synthetic-target-secret",
            timeout=0.5,
        )
        user = tai.FITDUser(
            chats["Target"],
            chats["Attacker"],
            Message("Explain the fictional puzzle"),
            tai.ContainsTextEvaluator("KEY-FOUND"),
            artifacts_dir=tmp_path,
        )
    user.start()
    try:
        assert user.wait(10)
        assert user.status().state is tai.UserState.FAILED
        assert error_name in user.status().reason
        assert user.result().to_string() == "Puzzle introduction"
        assert [
            turn.message.to_string()
            for turn in chats["Target"].history()
            if turn.role == "assistant"
        ] == ["Unrelated story", "Puzzle introduction"]
        events, metadata = artifacts(user, tmp_path)
        failures = [event["data"] for event in events if event["kind"] == "call_failed"]
        assert len(failures) == 1 and failures[0]["role"] == "Target"
        assert failures[0]["error"] == metadata["reason"]
        starts = [event["data"] for event in events if event["kind"] == "call_started"]
        assert failures[0]["call_id"] == starts[-1]["call_id"]
        assert len(starts) == len(endpoint.requests)
        assert (
            len([event for event in events if event["kind"] == "call_completed"])
            == len(starts) - 1
        )
        assert metadata["evaluation"] is None
        assert metadata["result"] == json.loads(user.result().dump())
        assert (tmp_path / user.status().run_id / "result.txt").read_text() == (
            "Puzzle introduction"
        )
        assert [
            event["data"]["result"]["text"]
            for event in events
            if event["kind"] == "result_published"
        ] == ["Puzzle introduction"]
        if reply.hold:
            assert not reply.release.is_set()
            assert reply.disconnected.wait(1)
    finally:
        reply.release.set()
        user.stop()
        assert user.wait(5)


def test_settings_redact_credentials_even_when_repeated_in_generation_metadata(
    endpoint,
):
    chat = OpenAIChat(
        base_url=endpoint.url,
        model="puzzle",
        api_key="synthetic-secret",
        generation={
            "metadata": {"label": "synthetic-secret", "api_key": "another-secret"}
        },
    )
    settings = chat.settings()
    assert "synthetic-secret" not in json.dumps(settings)
    assert "another-secret" not in json.dumps(settings)
    assert settings["generation"]["metadata"] == {
        "label": "[REDACTED]",
        "api_key": "[REDACTED]",
    }
    chat.send(Message("Puzzle clue"))
    assert endpoint.requests[0][2]["metadata"]["label"] == "synthetic-secret"


def test_external_providers_are_rejected_by_the_offline_guard_in_http_workers():
    chat = OpenAIChat(base_url="http://provider.invalid/v1", model="puzzle", timeout=2)
    with pytest.raises(tai.OpenAITransportError):
        chat.send(Message("Fictional clue"))
    assert chat.history() == ()


def test_truncated_body_preserves_observed_bytes_as_safe_diagnostics(endpoint):
    payload = "OBSERVED_PARTIAL π synthetic-secret".encode()
    endpoint.scripts["/v1/chat/completions"] = deque(
        [
            Reply("Saved hint"),
            Reply(payload=payload, truncated=True),
            Reply("Recovered"),
        ]
    )
    chat = OpenAIChat(base_url=endpoint.url, model="puzzle", api_key="synthetic-secret")
    chat.send(Message("First clue"))
    history = chat.history()
    with pytest.raises(tai.OpenAITransportError) as caught:
        chat.send(Message("Truncated clue"))
    error = caught.value
    assert error.response_bytes == len(payload)
    assert error.status_code == 200
    assert error.diagnostic == {
        "status_code": 200,
        "response_bytes": len(payload),
        "response_fragment": "OBSERVED_PARTIAL π [REDACTED]",
    }
    assert "synthetic-secret" not in str(error) + json.dumps(error.diagnostic)
    assert "IncompleteRead" in str(error)
    assert chat.history() == history
    assert chat.send(Message("Retry clue")).to_string() == "Recovered"
    assert "Truncated clue" not in json.dumps(endpoint.requests[-1][2]["messages"])


def test_real_fitd_retains_accepted_result_and_journals_truncated_body(
    endpoint, tmp_path
):
    payload = b"OBSERVED_PARTIAL KEY-FOUND synthetic-target-secret"
    user, chats = puzzle_user(
        endpoint,
        tmp_path,
        target_replies=[
            Reply(text) for text in ["One", "Two", "Three", "Four", "Accepted report"]
        ]
        + [Reply(payload=payload, truncated=True)],
        attacker_replies=[
            Reply(text) for text in ["Initial", "L1", "L2", "L3"] + ["yes"] * 5
        ],
    )
    user.start()
    assert user.wait(10)
    status, result = user.status(), user.result()
    assert status.state is tai.UserState.FAILED, status.reason
    assert "IncompleteRead" in status.reason
    assert f"response_bytes={len(payload)}" in status.reason
    assert result.to_string() == "Accepted report"
    assert status.evaluation.result_id == result.id
    assert status.evaluation.success is False
    assert status.progress.target_calls == 5
    assert chats["Target"].history()[-1].message is result
    assert len(chats["Target"].history()) == 11
    events, metadata = artifacts(user, tmp_path)
    failure = next(event["data"] for event in events if event["kind"] == "call_failed")
    assert failure["diagnostic"] == {
        "status_code": 200,
        "response_bytes": len(payload),
        "response_fragment": "OBSERVED_PARTIAL KEY-FOUND [REDACTED]",
    }
    starts = [event["data"] for event in events if event["kind"] == "call_started"]
    assert failure["role"] == "Target"
    assert failure["call_id"] == starts[-1]["call_id"]
    assert failure["error"] == status.reason
    completed = [event["data"] for event in events if event["kind"] == "call_completed"]
    assert all(event["call_id"] != failure["call_id"] for event in completed)
    assert len(completed) == len(starts) - 1
    assert all(
        "OBSERVED_PARTIAL" not in turn.message.to_string()
        for turn in chats["Target"].history()
    )
    assert not any(
        event["kind"] == "evaluation_completed" and event["data"]["success"]
        for event in events
    )
    assert metadata["result"] == json.loads(result.dump())
    assert metadata["evaluation"] == {"result_id": result.id, "success": False}
    assert (tmp_path / status.run_id / "result.txt").read_text() == "Accepted report"


def observe_partial_worker(monkeypatch, prefix):
    """Synchronize cancellation with actual pipe delivery, not a server-side write."""
    observed = Event()
    workers = []
    popen = subprocess.Popen
    read = os.read
    received = {}

    class ObservedWorker(popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            workers.append(self)
            received[self.stdout.fileno()] = bytearray()

    def observe_read(fd, size):
        chunk = read(fd, size)
        if fd in received:
            received[fd].extend(chunk)
            if prefix in received[fd]:
                observed.set()
        return chunk

    monkeypatch.setattr(subprocess, "Popen", ObservedWorker)
    monkeypatch.setattr(os, "read", observe_read)
    return observed, workers


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_interrupted_body_preserves_observed_diagnostics(endpoint, monkeypatch, stop):
    prefix = "OBSERVED_PARTIAL π synthetic-secret".encode()
    observed, workers = observe_partial_worker(monkeypatch, prefix)
    held = Reply(
        payload=prefix + b"UNSEEN_SUFFIX",
        hold=True,
        partial=True,
        partial_bytes=len(prefix),
    )
    endpoint.scripts["/v1/chat/completions"] = deque(
        [Reply("Saved hint"), held, Reply("Recovered")]
    )
    chat = OpenAIChat(
        base_url=endpoint.url,
        model="puzzle",
        api_key="synthetic-secret",
        timeout=1 if stop == "timeout" else 120,
    )
    chat.send(Message("First clue"))
    history = chat.history()
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(chat.send, Message("Held clue"))
        try:
            assert held.entered.wait(5)
            assert observed.wait(5), "worker did not report the received body prefix"
            if stop == "cancel":
                chat.cancel()
                expected = tai.ChatCancelled
            else:
                expected = tai.OpenAITimeoutError
            with pytest.raises(expected) as caught:
                pending.result(timeout=2)
            assert caught.value.diagnostic == {
                "status_code": 200,
                "response_bytes": len(prefix),
                "response_fragment": "OBSERVED_PARTIAL π [REDACTED]",
            }
            if stop == "cancel":
                assert caught.value.response is None
            else:
                assert caught.value.response_bytes == len(prefix)
            assert held.disconnected.wait(1)
            assert not held.release.is_set()
            assert chat.history() == history
            for worker in workers:
                assert worker.returncode is not None
                with pytest.raises(ChildProcessError):
                    os.waitpid(worker.pid, os.WNOHANG)
        finally:
            chat.cancel()
            held.release.set()
    assert chat.send(Message("Retry clue")).to_string() == "Recovered"
    assert "Held clue" not in json.dumps(endpoint.requests[-1][2]["messages"])


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_real_fitd_interrupted_body_is_only_a_diagnostic(
    endpoint, tmp_path, monkeypatch, stop
):
    prefix = b"OBSERVED_PARTIAL KEY-FOUND synthetic-target-secret"
    observed, workers = observe_partial_worker(monkeypatch, prefix)
    held = Reply(
        payload=prefix + b"UNSEEN_SUFFIX",
        hold=True,
        partial=True,
        partial_bytes=len(prefix),
    )
    user, chats = puzzle_user(
        endpoint,
        tmp_path,
        target_replies=[
            Reply(text) for text in ["One", "Two", "Three", "Four", "Accepted report"]
        ]
        + [held],
        attacker_replies=[
            Reply(text) for text in ["Initial", "L1", "L2", "L3"] + ["yes"] * 5
        ],
        target_timeout=1 if stop == "timeout" else 120,
    )
    user.start()
    try:
        assert held.entered.wait(5)
        assert observed.wait(5), "worker did not report the received body prefix"
        result = user.result()
        evaluation = user.status().evaluation
        histories = {role: chat.history() for role, chat in chats.items()}
        calls = len(endpoint.requests)
        if stop == "cancel":
            user.stop()
        assert user.wait(3)
        assert held.disconnected.wait(1)
        assert not held.release.is_set()
        assert user.status().state is (
            tai.UserState.STOPPED if stop == "cancel" else tai.UserState.FAILED
        )
        assert user.result() is result
        assert result.to_string() == "Accepted report"
        assert user.status().evaluation == evaluation
        assert evaluation.result_id == result.id and evaluation.success is False
        assert {role: chat.history() for role, chat in chats.items()} == histories
        assert len(endpoint.requests) == calls
        assert user.status().progress.target_calls == 5
        events, metadata = artifacts(user, tmp_path)
        kind = "call_cancelled" if stop == "cancel" else "call_failed"
        interrupted = next(event["data"] for event in events if event["kind"] == kind)
        assert interrupted["diagnostic"] == {
            "status_code": 200,
            "response_bytes": len(prefix),
            "response_fragment": "OBSERVED_PARTIAL KEY-FOUND [REDACTED]",
        }
        if stop == "cancel":
            assert interrupted["response"] is None
            assert user.status().reason == "user_stop"
        else:
            assert interrupted["error"] == user.status().reason
            assert "OpenAITimeoutError" in user.status().reason
        starts = [event["data"] for event in events if event["kind"] == "call_started"]
        assert interrupted["call_id"] == starts[-1]["call_id"]
        assert interrupted["role"] == "Target"
        assert not any(
            event["kind"] == "call_completed"
            and event["data"]["call_id"] == interrupted["call_id"]
            for event in events
        )
        assert "UNSEEN_SUFFIX" not in json.dumps(events)
        assert metadata["result"] == json.loads(result.dump())
        assert metadata["evaluation"] == {"result_id": result.id, "success": False}
        assert (tmp_path / user.status().run_id / "result.txt").read_text() == (
            "Accepted report"
        )
        for worker in workers:
            assert worker.returncode is not None
            with pytest.raises(ChildProcessError):
                os.waitpid(worker.pid, os.WNOHANG)
    finally:
        user.stop()
        held.release.set()
        assert user.wait(5)


def test_http_checkpoint_restore_and_reset_control_the_next_actual_request(endpoint):
    endpoint.scripts["/v1/chat/completions"] = deque(
        Reply(text) for text in ["First", "Discarded", "Replacement", "Fresh"]
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
        {"role": "system", "content": "Puzzle rules"},
        {"role": "user", "content": "Opening"},
        {"role": "assistant", "content": "First"},
        {"role": "user", "content": "Alternative clue"},
    ]
    chat.reset()
    with pytest.raises(ValueError, match="checkpoint"):
        chat.restore(point)
    chat.send(Message("New conversation"))
    assert endpoint.requests[3][2]["messages"] == [
        {"role": "system", "content": "Puzzle rules"},
        {"role": "user", "content": "New conversation"},
    ]


@pytest.mark.parametrize("first_outcome", ["COMPLETED", "STOPPED"])
def test_same_user_restarts_actual_http_chats_with_fresh_history_and_safe_settings(
    endpoint, tmp_path, first_outcome
):
    user, chats = puzzle_user(endpoint, tmp_path)
    held = Reply(hold=True)
    if first_outcome == "STOPPED":
        endpoint.scripts["/v1/target/chat/completions"][3] = held
    user.start()
    try:
        if first_outcome == "STOPPED":
            assert held.entered.wait(5)
            user.stop()
        assert user.wait(10)
        assert user.status().state.value == first_outcome, user.status().reason
        if first_outcome == "STOPPED":
            assert held.disconnected.wait(1)
            assert not held.release.is_set()
    finally:
        held.release.set()
        user.stop()
        assert user.wait(5)
    old_status, old_result = user.status(), user.result()
    old_dump = old_result.dump()
    old_directory = tmp_path / old_status.run_id
    old_files = {path.name: path.read_bytes() for path in old_directory.iterdir()}
    _, old_metadata = artifacts(user, tmp_path)
    points = {role: chat.checkpoint() for role, chat in chats.items()}
    offset = len(endpoint.requests)
    next_initial = Reply("New opening clue", hold=True)
    endpoint.scripts = {
        "/v1/attacker/chat/completions": deque(
            [next_initial] + [Reply(text) for text in ["L1", "L2", "L3"] + ["yes"] * 5]
        ),
        "/v1/target/chat/completions": deque(
            Reply(text)
            for text in [
                "New one",
                "New two",
                "New three",
                "New four",
                "KEY-FOUND new solution",
            ]
        ),
    }
    user.start()
    try:
        assert next_initial.entered.wait(5)
        assert user.status().run_id != old_status.run_id
        assert user.result() is None
        assert user.status().evaluation is None
        assert all(len(chat.history()) == 1 for chat in chats.values())
        with pytest.raises(ValueError, match="checkpoint"):
            chats["Target"].restore(points["Target"])
        # Run-entry reset and Attacker reset retain both ownership reservations.
        contender = tai.FITDUser(
            chats["Target"],
            tai.LLMChat(lambda turns: "unused"),
            Message("Other"),
            tai.ContainsTextEvaluator("KEY-FOUND"),
            artifacts_dir=tmp_path,
        )
        with pytest.raises(RuntimeError, match="LLMChat.*active"):
            contender.start()
    finally:
        next_initial.release.set()
        assert user.wait(10)
    assert user.status().state is tai.UserState.COMPLETED, user.status().reason
    assert user.status().reason == "evaluator_success"
    assert user.result().to_string() == "KEY-FOUND new solution"
    _, metadata = artifacts(user, tmp_path)
    assert metadata["settings"] == old_metadata["settings"]
    assert old_result.dump() == old_dump
    assert {
        path.name: path.read_bytes() for path in old_directory.iterdir()
    } == old_files
    for role, chat in chats.items():
        before = chat.history()
        with pytest.raises(ValueError, match="checkpoint"):
            chat.restore(points[role])
        assert chat.history() == before
        first = next(
            request
            for request in endpoint.requests[offset:]
            if f"/{role.lower()}/" in request[0]
        )
        assert [turn["role"] for turn in first[2]["messages"]] == ["system", "user"]
        assert first[2]["messages"][0]["content"] == (
            "Target puzzle keeper" if role == "Target" else "Attacker puzzle author"
        )
