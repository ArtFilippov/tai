"""Independent LLMChat with lazy server diagnostics and role generation settings."""

from collections.abc import Mapping
from threading import Event
from typing import override

from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.messages import Message

from ..chat_turn_contract import ChatTurn
from ..server_contract import Server
from .conversation import LLMChat
from .server_validation_utils import http_generation

type PreparedRole = tuple[str, DiagnosticRedactor, str, Mapping[str, object], float]


class QwenServerChat(LLMChat):
    def __init__(
        self, server: Server, *, initial: Message | None, role: PreparedRole,
    ) -> None:
        super().__init__(
            lambda turns: "", initial=initial, model=role[0], system=role[2], redactor=role[1],
        )
        self.server = server
        self.parameters = dict(role[3])
        self.timeout: float = role[4]

    @override
    def __call__(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        return self.server.generate(
            turns, cancellation, generation=self.parameters, timeout=self.timeout,
        )

    @override
    def settings(self) -> dict[str, object] | list[object]:
        return self.diagnostic_settings(self.redactor)

    @override
    def diagnostic_settings(
        self, redactor: DiagnosticRedactor,
    ) -> dict[str, object] | list[object]:
        return redactor.structured({
            **super().settings_data(), "backend": "vllm", "server": self.server.settings(),
            "timeout": self.timeout, "generation": http_generation(self.parameters),
        })
