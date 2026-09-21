"""Backpressure, cancellation and deterministic cleanup at the real OS boundary."""

import json
import signal
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack

import pytest
from chats_assertions import HttpEndpoint
from http_peer import Reply
from openai_workers import failing_worker, gate_worker_input, observe_partial_worker, observe_workers

import tai
from tai import Message, OpenAIChat


def interrupt(
    chat: OpenAIChat, pending: Future[Message], stop: str,
) -> tai.ChatCancelled | tai.OpenAITimeoutError:
    if stop == "cancel":
        chat.cancel()
        expected = tai.ChatCancelled
    else:
        expected = tai.OpenAITimeoutError
    with pytest.raises(expected) as caught:
        pending.result(timeout=2)
    return caught.value


def _large_request(
    endpoint: HttpEndpoint, chat: OpenAIChat, source: str,
) -> tuple[Message, list[dict[str, str]]]:
    initial = "Question:\n" + "q" * 150000 + "\nEND π"
    assistant = "Answer:\n" + "a" * 150054 + "\nEND π"
    if source == "assistant_history":
        endpoint.scripts["/v1/chat/completions"] = deque([Reply(assistant), Reply("Done")])
        assert chat.send(Message("Opening")).to_string() == assistant
        return Message("Next clue"), [
            {"role": "system", "content": "Puzzle rules"},
            {"role": "user", "content": "Opening"},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": "Next clue"},
        ]
    endpoint.scripts["/v1/chat/completions"] = deque([Reply("Done")])
    return Message(initial), [
        {"role": "system", "content": "Puzzle rules"}, {"role": "user", "content": initial},
    ]


@pytest.mark.parametrize("large_source", ["initial", "assistant_history"])
def test_large_request_reaches_endpoint_after_stdin_backpressure(
    endpoint: HttpEndpoint, monkeypatch: pytest.MonkeyPatch, large_source: str,
) -> None:
    chat = OpenAIChat(base_url=endpoint.url, model="puzzle", system="Puzzle rules", timeout=3)
    message, expected = _large_request(endpoint, chat, large_source)
    previous = chat.history()
    calls = len(endpoint.requests)
    filled, workers = gate_worker_input(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(chat.send, message)
        with ExitStack() as cleanup:
            cleanup.callback(workers.cancel, chat)
            assert filled.wait(2), "large request did not fill the stopped worker's pipe"
            # The child cannot read; pending spans actual control polling cycles.
            with pytest.raises(TimeoutError):
                pending.result(timeout=0.1)
            assert chat.history() == previous
            assert len(endpoint.requests) == calls
            workers.processes[0].send_signal(signal.SIGCONT)
            assert pending.result(timeout=5).to_string() == "Done"
            assert endpoint.requests[-1][2]["messages"] == expected
            assert len(endpoint.requests) == calls + 1
            assert chat.history()[:-2] == previous
            assert chat.history()[-2].message is message
    workers.assert_reaped()


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_large_blocked_input_can_be_interrupted_and_all_local_io_is_joined(
    endpoint: HttpEndpoint, monkeypatch: pytest.MonkeyPatch, stop: str,
) -> None:
    endpoint.scripts["/v1/chat/completions"] = deque([Reply("Saved"), Reply("Recovered")])
    chat = OpenAIChat(base_url=endpoint.url, model="puzzle", timeout=0.5 if stop == "timeout" else 120)
    chat.send(Message("Opening"))
    previous = chat.history()
    with monkeypatch.context() as patch:
        filled, workers = gate_worker_input(patch)
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = executor.submit(chat.send, Message("x" * 150000))
            with ExitStack() as cleanup:
                cleanup.callback(workers.cancel, chat)
                assert filled.wait(2)
                assert not pending.done()
                assert len(endpoint.requests) == 1
                if stop == "cancel":
                    assert executor.submit(chat.cancel).result(timeout=1) is None
                error = interrupt(chat, pending, stop)
                assert error.diagnostic is None
                assert chat.history() == previous
                assert len(endpoint.requests) == 1
                # No SIGCONT: kill must end a process blocked on stdin delivery.
                assert len(workers.communicators) == 1
                workers.assert_reaped(pipes=True)
    assert chat.send(Message("Retry")).to_string() == "Recovered"
    assert endpoint.requests[-1][2]["messages"] == [
        {"role": "user", "content": "Opening"}, {"role": "assistant", "content": "Saved"},
        {"role": "user", "content": "Retry"},
    ]


def test_communication_error_joins_io_thread_and_reaps_worker(
    endpoint: HttpEndpoint, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers = failing_worker(monkeypatch)
    chat = OpenAIChat(base_url=endpoint.url, model="puzzle")
    with pytest.raises(tai.OpenAITransportError, match="worker communication failed"):
        chat.send(Message("Clue"))
    assert not endpoint.requests
    assert chat.history() == ()
    assert len(workers.communicators) == 1
    workers.assert_reaped(pipes=True)


def _assert_active_history(chat: OpenAIChat, point: object) -> None:
    for mutate in (chat.reset, lambda: chat.restore(point)):
        with pytest.raises(RuntimeError, match="active exchange"):
            mutate()


@pytest.mark.parametrize("partial", [False, True], ids=["headers", "body"])
@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_interruption_closes_local_io_before_the_server_releases_a_response(
    endpoint: HttpEndpoint, partial: bool, stop: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers = observe_workers(monkeypatch)
    held = Reply(hold=True, partial=partial)
    endpoint.scripts["/v1/chat/completions"] = deque([Reply("Saved hint"), held, Reply("")])
    chat = OpenAIChat(base_url=endpoint.url, model="puzzle", timeout=0.5 if stop == "timeout" else 120)
    chat.cancel()  # Idle cancellation must not poison the next exchange.
    chat.send(Message("First clue"))
    history, point = chat.history(), chat.checkpoint()
    with ThreadPoolExecutor(max_workers=1) as worker:
        pending = worker.submit(chat.send, Message("Held clue"))
        with ExitStack() as cleanup:
            cleanup.callback(held.release.set)
            assert held.entered.wait(5)
            _assert_active_history(chat, point)
            error = interrupt(chat, pending, stop)
            if not partial:
                assert error.diagnostic is None
            assert not held.release.is_set()
            assert held.disconnected.wait(1), "local socket leaked after send unwound"
            assert chat.history() == history
            assert len(workers.processes) == 2
            workers.assert_reaped()
            chat.restore(point)
            assert chat.history() == history
    chat.cancel()
    assert not chat.send(Message("Fresh clue")).to_string()
    assert len(endpoint.requests) == 3
    assert "Held clue" not in json.dumps(endpoint.requests[-1][2])


def _assert_partial_error(error: tai.ChatCancelled | tai.OpenAITimeoutError, prefix: bytes) -> None:
    assert error.diagnostic == {
        "status_code": 200, "response_bytes": len(prefix),
        "response_fragment": "OBSERVED_PARTIAL π [REDACTED]",
    }
    if isinstance(error, tai.ChatCancelled):
        assert error.response is None
    else:
        assert error.response_bytes == len(prefix)


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_interrupted_body_preserves_observed_diagnostics(
    endpoint: HttpEndpoint, monkeypatch: pytest.MonkeyPatch, stop: str,
) -> None:
    prefix = "OBSERVED_PARTIAL π synthetic-secret".encode()
    observed, workers = observe_partial_worker(monkeypatch, prefix)
    held = Reply(payload=prefix + b"UNSEEN_SUFFIX", hold=True, partial=True, partial_bytes=len(prefix))
    endpoint.scripts["/v1/chat/completions"] = deque([Reply("Saved hint"), held, Reply("Recovered")])
    chat = OpenAIChat(
        base_url=endpoint.url, model="puzzle", api_key="synthetic-secret",
        timeout=1 if stop == "timeout" else 120,
    )
    chat.send(Message("First clue"))
    history = chat.history()
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(chat.send, Message("Held clue"))
        with ExitStack() as cleanup:
            cleanup.callback(held.release.set)
            cleanup.callback(chat.cancel)
            assert held.entered.wait(5)
            assert observed.wait(5), "worker did not report the received body prefix"
            _assert_partial_error(interrupt(chat, pending, stop), prefix)
            assert held.disconnected.wait(1)
            assert not held.release.is_set()
            assert chat.history() == history
            workers.assert_reaped()
    assert chat.send(Message("Retry clue")).to_string() == "Recovered"
    assert "Held clue" not in json.dumps(endpoint.requests[-1][2]["messages"])
