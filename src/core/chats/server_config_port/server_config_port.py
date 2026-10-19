"""Pinned server profile passed to the managed process without GPU imports."""

from pathlib import Path
from typing import Protocol

from core.chats.server_limits_port.server_limits_port import ServerLimitsPort


class ServerConfigPort(Protocol):
    """Describe the already prepared model and its resource limits."""

    def snapshot(self) -> Path: ...

    def device(self) -> str: ...

    def interpreter(self) -> Path: ...

    def precision(self) -> str: ...

    def resources(self) -> ServerLimitsPort: ...

    def version(self) -> str: ...
