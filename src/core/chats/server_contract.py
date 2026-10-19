"""Shared generation resource without ownership of an individual conversation."""

from collections.abc import Mapping
from threading import Event
from typing import Protocol

from core.primitives.diagnostics import DiagnosticRedactor

from .chat_turn_contract import ChatTurn


class Server(Protocol):
    """Generate for independent chats and release the common model process."""

    def model(self) -> str: ...

    def settings(self) -> dict[str, object]: ...

    def redactor(self, token: str = "") -> DiagnosticRedactor: ...

    def generate(
        self, turns: tuple[ChatTurn, ...], cancellation: Event, *,
        generation: Mapping[str, object] | None = None, timeout: float = 120.0,
    ) -> str: ...

    def close(self) -> None: ...
