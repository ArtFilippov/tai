"""Lifecycle and authenticated identity of one owned model process."""

from threading import Event
from typing import Protocol


class ServerProcessPort(Protocol):
    """Expose process readiness and identity without sharing conversation state."""

    def identity(self) -> tuple[str, str, str]: ...

    def status(self) -> dict[str, object]: ...

    def launch(self) -> None: ...

    def check_alive(self) -> None: ...

    def await_ready(self, closing: Event) -> None: ...

    def close(self) -> None: ...
