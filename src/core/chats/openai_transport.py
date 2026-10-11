"""Disposable-worker exchange lifecycle and typed completion decoding."""

import json
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
from threading import Event
from time import monotonic
from typing import cast

from core.primitives.diagnostics import DiagnosticRedactor

from .envelopes.cancelled import Cancelled
from .http_contracts import Interruption, WorkerResult
from .openai_errors import (
    OpenAIAuthenticationError, OpenAIContentError, OpenAIHTTPError,
    OpenAIJSONError, OpenAIRateLimitError, OpenAITimeoutError, OpenAITransportError,
)

_ERROR_FIELD = "error"
_ASSISTANT_ROLE = "assistant"


def _worker() -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("http_worker.py"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            # Ctrl+C belongs to User.stop(), not the disposable transport child.
            start_new_session=True,
        )
    except OSError:
        raise OpenAITransportError("could not start HTTP worker") from None


def _wait(
    output: Future[tuple[bytes, bytes]], cancellation: Event, deadline: float,
) -> Interruption | None:
    for _ in iter(output.done, True):
        if cancellation.is_set():
            return Interruption.CANCEL
        if (remaining := deadline - monotonic()) <= 0:
            return Interruption.TIMEOUT
        _ = cancellation.wait(min(0.02, remaining))
    return None


def exchange(request: bytes, cancellation: Event, timeout: float) -> WorkerResult:
    deadline = monotonic() + timeout
    with _worker() as worker, ThreadPoolExecutor(max_workers=1, thread_name_prefix="tai-http-io") as io:
        # One communicate call owns all stdin writes and drains both output pipes.
        output = io.submit(worker.communicate, request)
        try:
            interrupted = _wait(output, cancellation, deadline)
        finally:
            if worker.poll() is None:
                worker.kill()
            # Killing releases blocked writes; collect both streams before leaving.
            _ = worker.wait()
        return _collect_output(output, worker.returncode, interrupted)


def _collect_output(
    output: Future[tuple[bytes, bytes]], returncode: int | None,
    interrupted: Interruption | None,
) -> WorkerResult:
    failed = output.exception() is not None
    raw, reports = (b"", b"") if failed else output.result()
    return WorkerResult(raw, reports, returncode, interrupted, failed)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    return cast(dict[str, object], value)


def _reports(raw: bytes) -> dict[str, object]:
    response: dict[str, object] = {}
    for line in raw.splitlines():
        try:
            record = _mapping(json.loads(line))
        except ValueError:
            # A kill can interrupt status writes. Never surface stderr text.
            continue
        response.update(record)
    return response


def _failure(result: WorkerResult, response: dict[str, object]) -> str | None:
    if result.communication_failed:
        return "worker communication failed"
    failure = response.get("error")
    if isinstance(failure, str):
        return failure
    if result.returncode or not response.get("complete") or response.get("status") is None:
        return "worker failed"
    return None


def _check_transport(
    result: WorkerResult, response: dict[str, object], redactor: DiagnosticRedactor,
) -> int:
    failure = _failure(result, response)
    status = response.get("status")
    status = status if isinstance(status, int) else None
    fragment = redactor.text(result.text(errors="replace")) if result.raw else None
    if result.interrupted is Interruption.CANCEL:
        raise Cancelled(diagnostic={
            "status_code": status, "response_bytes": len(result.raw),
            "response_fragment": fragment,
        } if result.raw else None)
    if result.interrupted is Interruption.TIMEOUT or failure == TimeoutError.__name__:
        raise OpenAITimeoutError(
            "request timed out", status_code=status,
            response_bytes=len(result.raw), response_fragment=fragment,
        )
    if failure is not None:
        raise OpenAITransportError(
            f"transport failed: {failure}", status_code=status,
            response_bytes=len(result.raw), response_fragment=fragment,
        )
    assert status is not None
    return status


def _check_status(status: int) -> None:
    if HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
        return
    error_types: dict[int, type[OpenAIHTTPError]] = {
        HTTPStatus.UNAUTHORIZED: OpenAIAuthenticationError,
        HTTPStatus.FORBIDDEN: OpenAIAuthenticationError,
        HTTPStatus.TOO_MANY_REQUESTS: OpenAIRateLimitError,
    }
    error_type = error_types.get(status, OpenAIHTTPError)
    raise error_type("HTTP request rejected", status_code=status)


def _content(value: object) -> str:
    body = _mapping(value)
    choices = body["choices"]
    if not isinstance(choices, list) or _ERROR_FIELD in body:
        raise ValueError("expected completion choices")
    entries = cast(list[object], choices)
    if len(entries) != 1:
        raise ValueError("expected one choice")
    message = _mapping(_mapping(entries[0])["message"])
    content = message["content"]
    if message["role"] != _ASSISTANT_ROLE or not isinstance(content, str):
        raise ValueError("expected assistant text")
    return content


def completion(result: WorkerResult, redactor: DiagnosticRedactor) -> str:
    status = _check_transport(result, _reports(result.reports), redactor)
    _check_status(status)
    try:
        body: object = json.loads(result.text())
    except ValueError:
        raise OpenAIJSONError(
            "invalid JSON", status_code=status, response_bytes=len(result.raw),
        ) from None
    try:
        return _content(body)
    except (KeyError, IndexError, TypeError, ValueError):
        raise OpenAIContentError(
            "expected one assistant text completion", status_code=status,
            response_bytes=len(result.raw),
        ) from None
