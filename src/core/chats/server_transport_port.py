"""One cancellable HTTP request with detached response diagnostics."""

from threading import Event
from typing import Protocol

from core.primitives.diagnostics import DiagnosticRedactor


class ServerTransportPort(Protocol):
    """Exchange a complete request without retaining a chat's history."""

    def __call__(
        self, data: bytes, cancellation: Event, timeout: float, redactor: DiagnosticRedactor,
    ) -> str: ...
