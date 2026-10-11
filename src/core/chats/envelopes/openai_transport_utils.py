"""Assemble HTTP exchange and failure translation for OpenAI-compatible chats."""

from threading import Event
from typing import Literal

from core.primitives.diagnostics import DiagnosticRedactor

from ..openai_transport_engine_utils import completion as decode
from ..openai_transport_engine_utils import exchange as request
from .openai_error_utils import failure

type WorkerResult = tuple[bytes, bytes, int | None, Literal["cancel", "timeout"] | None, bool]


def exchange(data: bytes, cancellation: Event, timeout: float) -> WorkerResult:
    return request(data, cancellation, timeout, failure)


def completion(result: WorkerResult, redactor: DiagnosticRedactor) -> str:
    return decode(result, redactor, failure)
