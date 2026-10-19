"""A lazy shared generation resource; conversations remain in independent chats."""

from collections.abc import Mapping
from threading import Event, Lock
from types import TracebackType
from typing import Self, override

from core.primitives.diagnostics import DiagnosticRedactor, RedactionFactory

from ..server_lifecycle import ServerLifecycle
from ..server_request_utils import request
from ..chat_turn_contract import ChatTurn
from ..server_contract import Server
from ..server_ports import ServerProcessPort
from ..server_transport_port import ServerTransportPort
from .cancelled import Cancelled
from .server_config import ServerConfig
from .server_validation_utils import http_generation, positive_number
from .server_admission_utils import admitted
from .qwen_server_error import QwenServerError
from .server_transport_utils import complete

__all__ = ["QwenServer", "QwenServerError"]


class QwenServer(Server):
    def __init__(
        self, config: ServerConfig, redaction: RedactionFactory, process: ServerProcessPort,
        transport: ServerTransportPort = complete,
    ) -> None:
        super().__init__()
        self.lifecycle = ServerLifecycle()
        self.config, self.process = config, process
        self.lock: Lock = Lock()
        self.redaction: RedactionFactory = redaction
        self.transport: ServerTransportPort = transport

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @override
    def settings(self) -> dict[str, object]:
        with self.lifecycle.condition:
            return {
                **self.config.settings(), "state": self.lifecycle.state,
                **self.process.status(),
            }

    @override
    def redactor(self, token: str = "") -> DiagnosticRedactor:
        return self.redaction((token,))

    @override
    def model(self) -> str:
        return self.config.checkpoint.key(self.config.dtype)[0]

    def start(self) -> None:
        try:
            self.launch()
        except (OSError, ValueError, QwenServerError) as exc:
            self.process.close()
            with self.lifecycle.condition:
                self.lifecycle.state = "failed"
            self.lifecycle.failed(exc)
        else:
            self.lifecycle.started()

    def launch(self) -> None:
        self.process.launch()
        self.process.await_ready(self.lifecycle.closing)
        with self.lifecycle.condition:
            self.lifecycle.state = "ready"

    def ensure_ready(self, cancellation: Event) -> None:
        self.lifecycle.start(self.start)
        for _ in iter(self.lifecycle.wait_ready, True):
            if cancellation.is_set():
                raise Cancelled()
        _ = self.lifecycle.ready.result()
        self.check_alive()

    def check_alive(self) -> None:
        try:
            self.process.check_alive()
        except QwenServerError:
            with self.lifecycle.condition:
                self.lifecycle.state = "failed"
            raise

    @override
    def generate(
        self, turns: tuple[ChatTurn, ...], cancellation: Event, *,
        generation: Mapping[str, object] | None = None, timeout: float = 120.0,
    ) -> str:
        parameters = http_generation(generation)
        _ = positive_number(timeout, "request timeout")
        if cancellation.is_set():
            raise Cancelled()
        with admitted(self.lifecycle):
            self.ensure_ready(cancellation)
            if cancellation.is_set():
                raise Cancelled()
            response = self.transport(
                request(turns, parameters, timeout, self.process.identity()), cancellation, timeout,
                self.redactor(self.process.identity()[2]),
            )
            self.check_alive()
            return response

    @override
    def close(self) -> None:
        self.lifecycle.request_close()
        with self.lock:
            self.lifecycle.join()
            self.process.close()
            with self.lifecycle.condition:
                self.lifecycle.state = "closed"
