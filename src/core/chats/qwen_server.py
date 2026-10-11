"""A lazy shared generation resource; conversations remain in independent chats."""

import json
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from types import TracebackType
from typing import Self, override

from core.primitives.diagnostics import DiagnosticRedactor, RedactionFactory

from .conversation import ChatCancelled, ChatTurn
from .openai_transport import completion, exchange
from .server_config import ServerConfig, http_generation, positive_number
from .server_process import QwenServerError, ServerProcess
from .server_ports import Server, ServerProcessPort

__all__ = ["QwenServer", "QwenServerError"]
NOT_STARTED = "not_started"


@dataclass
class _Lifecycle:
    condition: Condition = field(default_factory=Condition)
    ready: Event = field(default_factory=Event)
    closing: Event = field(default_factory=Event)
    state: str = NOT_STARTED
    error: Exception | None = None
    thread: Thread | None = None
    active: int = 0

    def start(self, invoke: Callable[[], None]) -> None:
        with self.condition:
            if self.state == NOT_STARTED:
                self.state = "starting"
                self.thread = Thread(target=invoke, name="tai-vllm-start")
                self.thread.start()

    def started(self) -> None:
        self.ready.set()

    def wait_ready(self) -> bool:
        return self.ready.wait(0.02)

    def admit(self) -> None:
        with self.condition:
            if self.closing.is_set():
                raise QwenServerError("vLLM server is closed")
            self.active += 1

    def release(self) -> None:
        with self.condition:
            self.active -= 1
            self.condition.notify_all()

    def request_close(self) -> None:
        self.closing.set()

    def join(self) -> None:
        with self.condition:
            _ = self.condition.wait_for(lambda: not self.active)
        if self.thread is not None:
            self.thread.join()


class QwenServer(Server):
    def __init__(
        self, config: ServerConfig, log_path: Path,
        redaction: RedactionFactory,
    ) -> None:
        super().__init__()
        self.config = config
        self._process: ServerProcessPort = ServerProcess(config, log_path)
        self._lifecycle = _Lifecycle()
        self._close_lock = Lock()
        self.redaction: RedactionFactory = redaction

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, _kind: type[BaseException] | None, _value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    @override
    def settings(self) -> dict[str, object]:
        with self._lifecycle.condition:
            return {
                **self.config.settings(), "state": self._lifecycle.state,
                "version": self._process.version, "log_path": str(self._process.log_path),
                "pid": self._process.pid,
            }

    @override
    def redactor(self, token: str = "") -> DiagnosticRedactor:
        return self.redaction((token,))

    def _start(self) -> None:
        try:
            self._launch()
        except (OSError, ValueError, QwenServerError) as exc:
            self._process.close()
            with self._lifecycle.condition:
                self._lifecycle.error = exc
                self._lifecycle.state = "failed"
        finally:
            self._lifecycle.started()

    def _launch(self) -> None:
        self._process.launch()
        self._process.await_ready(self._lifecycle.closing)
        with self._lifecycle.condition:
            self._lifecycle.state = "ready"

    def _ensure_ready(self, cancellation: Event) -> None:
        self._lifecycle.start(self._start)
        for _ in iter(self._lifecycle.wait_ready, True):
            if cancellation.is_set():
                raise ChatCancelled()
        if self._lifecycle.error is not None:
            raise self._lifecycle.error
        self._check_alive()

    def _check_alive(self) -> None:
        try:
            self._process.check_alive()
        except QwenServerError as exc:
            with self._lifecycle.condition:
                self._lifecycle.error = exc
                self._lifecycle.state = "failed"
            raise

    @contextmanager
    def _admitted(self) -> Generator[None]:
        self._lifecycle.admit()
        try:
            yield
        finally:
            self._lifecycle.release()

    def _request(self, turns: tuple[ChatTurn, ...], parameters: dict[str, object], timeout: float) -> bytes:
        return json.dumps({
            "url": self._process.url + "/v1/chat/completions", "timeout": timeout,
            "headers": {"Authorization": "Bearer " + self._process.token, "Content-Type": "application/json"},
            "body": {**parameters, "model": self._process.model,
                     "messages": [{"role": turn.role, "content": turn.text()} for turn in turns]},
        }).encode()

    @override
    def generate(
        self, turns: tuple[ChatTurn, ...], cancellation: Event, *,
        generation: Mapping[str, object] | None = None, timeout: float = 120.0,
    ) -> str:
        parameters = http_generation(generation)
        _ = positive_number(timeout, "request timeout")
        if cancellation.is_set():
            raise ChatCancelled()
        with self._admitted():
            self._ensure_ready(cancellation)
            if cancellation.is_set():
                raise ChatCancelled()
            response = exchange(self._request(turns, parameters, timeout), cancellation, timeout)
            self._check_alive()
            return completion(response, self.redactor(self._process.token))

    @override
    def close(self) -> None:
        self._lifecycle.request_close()
        with self._close_lock:
            self._lifecycle.join()
            self._process.close()
            with self._lifecycle.condition:
                self._lifecycle.state = "closed"
