"""Independent LLMChat with lazy server diagnostics and role generation settings."""

from collections.abc import Mapping
from threading import Event
from typing import override

from core.primitives.diagnostics import DiagnosticObject, DiagnosticRedactor

from .conversation import ChatTurn, LLMChat
from .qwen_server import QwenServer
from .server_config import http_generation, positive_number


class QwenServerChat(LLMChat):
    def __init__(
        self, server: QwenServer, *, system: str = "",
        generation: Mapping[str, object] | None = None, timeout: float = 120.0,
    ) -> None:
        super().__init__(lambda turns: "", model=server.config.checkpoint.model, system=system)
        self._server = server
        self._generation = dict(generation or {})
        self._timeout = positive_number(timeout, "request timeout")
        _ = http_generation(self._generation)

    @override
    def _generate(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        return self._server.generate(turns, cancellation, generation=self._generation, timeout=self._timeout)

    @override
    def settings(self) -> DiagnosticObject:
        return self._diagnostic_settings(self.redactor)

    @override
    def _diagnostic_settings(self, redactor: DiagnosticRedactor) -> DiagnosticObject:
        return redactor.snapshot({
            **super()._settings(), "backend": "vllm", "server": self._server.settings(),
            "timeout": self._timeout, "generation": http_generation(self._generation),
        })
