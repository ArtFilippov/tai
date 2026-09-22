"""Lazy startup, concurrent HTTP, cancellation, errors, and process-group cleanup."""

import json
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from time import monotonic

import pytest
from journal_assertions import at

from core.chats import ChatCancelled, LLMChat
from core.chats.openai import OpenAIContentError, OpenAITimeoutError
from core.chats.qwen_config import local_checkpoint
from core.chats.qwen_server import QwenServer, QwenServerError
from core.chats.server_chat import QwenServerChat
from core.chats.server_config import ServerConfig, ServerLimits, http_generation
from core.chats.server_process import ServerProcess
from core.primitives.messages import Message


def wait_file(path: Path) -> None:
    deadline = monotonic() + 5
    for _ in iter(path.exists, True):
        assert monotonic() < deadline, str(path)
        Event().wait(0.01)


def alive(pid: int) -> bool:
    status = Path(f"/proc/{pid}/stat")
    return status.exists() and status.read_text(encoding="utf-8").split()[2] != "Z"


@pytest.fixture
def managed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[QwenServer, Path]:
    checkpoint = local_checkpoint("Qwen/test", "a" * 40, tmp_path, "cuda:0")
    checkpoint.snapshot.mkdir(parents=True)
    config = ServerConfig(checkpoint, Path(sys.executable), ServerLimits(startup_timeout=2, shutdown_timeout=0.1))
    original = ServerProcess._command

    def command(process: ServerProcess) -> list[str]:
        return [sys.executable, str(Path(__file__).with_name("vllm_peer.py")), *original(process)[3:]]

    monkeypatch.setattr(ServerProcess, "_command", command)
    return QwenServer(config, tmp_path / "vllm.log"), checkpoint.snapshot


def test_lazy_first_requests_overlap_with_one_server_and_independent_histories(managed: tuple[QwenServer, Path]) -> None:
    server, root = managed
    (root / "hold").touch()
    chats = [QwenServerChat(server, system=f"role-{i}") for i in range(3)]
    for chat in chats:
        assert at(chat.settings(), "server", "state") == "not_started"
    assert not (root / "launches").exists()
    with server, ThreadPoolExecutor(max_workers=3) as pool:
        results = [pool.submit(chat.send, Message(f"request-{i}")) for i, chat in enumerate(chats)]
        try:
            for i in range(3):
                wait_file(root / f"request-{i}.request")
            assert not any(result.done() for result in results)
        finally:
            for i in range(3):
                (root / f"request-{i}.release").touch()
        assert len((root / "launches").read_text().splitlines()) == 1
        assert [result.result(5).text for result in results] == [f"request-{i} reply" for i in range(3)]
        pid = int((root / "launches").read_text())
        for i, chat in enumerate(chats):
            request = json.loads((root / f"request-{i}.request").read_text())
            assert request["messages"] == [{"role": "system", "content": f"role-{i}"}, {"role": "user", "content": f"request-{i}"}]
            assert request["n"] == 1 and not request["temperature"]
            assert chat.history()[-1].text() == f"request-{i} reply"
    server.close()
    assert server.settings()["state"] == "closed" and not alive(pid)
    assert not alive(int((root / "child").read_text()))


def test_cancel_one_exchange_keeps_other_request_and_server_alive(managed: tuple[QwenServer, Path]) -> None:
    server, root = managed
    (root / "hold").touch()
    left, right = LLMChat.cancellable(server.generate), LLMChat.cancellable(server.generate)
    with server, ThreadPoolExecutor(max_workers=2) as pool:
        cancelled, completed = pool.submit(left.send, Message("left")), pool.submit(right.send, Message("right"))
        def check_cancel() -> None:
            left.cancel()
            with pytest.raises(ChatCancelled):
                cancelled.result(5)
            assert left.history() == () and not completed.done()
            (root / "right.release").touch()
            assert completed.result(5).text == "right reply"
            assert server.settings()["state"] == "ready"

        try:
            wait_file(root / "left.request")
            wait_file(root / "right.request")
            check_cancel()
        finally:
            (root / "left.release").touch()
            (root / "right.release").touch()


@pytest.mark.parametrize(("mode", "reason"), [("exit", "exited"), ("startup_hold", "timed out"), ("foreign", "unexpected model")])
def test_startup_failure_shared_and_partial_process_cleaned(managed: tuple[QwenServer, Path], mode: str, reason: str) -> None:
    server, root = managed
    (root / mode).touch()
    with server, ThreadPoolExecutor(max_workers=3) as pool:
        results = [pool.submit(server.generate, (), Event()) for _ in range(3)]
        errors = [result.exception(5) for result in results]
        assert all(error is errors[0] for error in errors)
        assert isinstance(errors[0], QwenServerError) and reason in str(errors[0])
        assert server.settings()["state"] == "failed"
        assert len((root / "launches").read_text().splitlines()) == 1
    assert not alive(int((root / "launches").read_text()))
    if (root / "child").exists():
        assert not alive(int((root / "child").read_text()))


def test_invalid_response_timeout_and_running_server_death(managed: tuple[QwenServer, Path]) -> None:
    server, root = managed
    (root / "invalid").touch()
    chat = QwenServerChat(server)
    with server:
        with pytest.raises(OpenAIContentError):
            chat.send(Message("invalid"))
        (root / "invalid").unlink()
        (root / "hold").touch()
        short = QwenServerChat(server, timeout=0.1)
        with pytest.raises(OpenAITimeoutError):
            short.send(Message("timeout"))
        (root / "timeout.release").touch()
        (root / "hold").unlink()
        assert chat.send(Message("recovered")).text == "recovered reply"
        os.kill(int((root / "launches").read_text()), signal.SIGKILL)
        deadline = monotonic() + 5
        for _ in iter(lambda: alive(int((root / "launches").read_text())), False):
            assert monotonic() < deadline
            Event().wait(0.01)
        with pytest.raises(QwenServerError, match="exited"):
            chat.send(Message("dead"))
        assert server.settings()["state"] == "failed"


def test_cancelled_admission_and_invalid_parameters_never_start(managed: tuple[QwenServer, Path]) -> None:
    server, root = managed
    signal_event = Event()
    signal_event.set()
    with server:
        with pytest.raises(ChatCancelled):
            server.generate((), signal_event)
        with pytest.raises(ValueError):
            QwenServerChat(server, generation={"max_new_tokens": -1})
        with pytest.raises(ValueError):
            replace(server.config, limits=ServerLimits(max_num_seqs=0))
    assert not (root / "launches").exists()
    assert http_generation({"do_sample": True, "temperature": 0.7, "top_k": 0})["top_k"] == -1
