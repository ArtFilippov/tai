"""Shared generation and owned-process lifecycle contracts, without GPU imports."""

from collections.abc import Mapping
from pathlib import Path
from threading import Event
from typing import Protocol

from .chat_state import ChatTurn
from .server_config import ServerConfig


class Server(Protocol):
    config: ServerConfig

    def settings(self) -> dict[str, object]: ...

    def generate(
        self, turns: tuple[ChatTurn, ...], cancellation: Event, *,
        generation: Mapping[str, object] | None = None, timeout: float = 120.0,
    ) -> str: ...

    def close(self) -> None: ...


class ServerProcessPort(Protocol):
    log_path: Path
    model: str
    token: str
    version: str | None

    @property
    def url(self) -> str: ...

    @property
    def pid(self) -> int | None: ...

    def launch(self) -> None: ...

    def check_alive(self) -> None: ...

    def await_ready(self, closing: Event) -> None: ...

    def close(self) -> None: ...
