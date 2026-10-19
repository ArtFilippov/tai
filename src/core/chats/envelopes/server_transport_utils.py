"""Complete an authenticated vLLM request through the disposable HTTP worker."""

from threading import Event

from core.primitives.diagnostics import DiagnosticRedactor

from .openai_transport_utils import completion, exchange


def complete(data: bytes, cancellation: Event, timeout: float, redactor: DiagnosticRedactor) -> str:
    return completion(exchange(data, cancellation, timeout), redactor)
