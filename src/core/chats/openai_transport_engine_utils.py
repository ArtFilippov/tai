"""Disposable-worker exchange lifecycle and typed completion decoding."""

import json
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Literal, cast

from core.primitives.diagnostics import DiagnosticRedactor

ERROR_FIELD = "error"
ASSISTANT_ROLE = "assistant"
type Interruption = Literal["cancel", "timeout"]
type WorkerResult = tuple[bytes, bytes, int | None, Interruption | None, bool]
type Failure = Callable[[str, int | None, int, str | None], Exception]


def require(valid: bool, message: str) -> None:
    if not valid:
        raise ValueError(message)


def worker(errors: Failure) -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).parent / "main" / "http_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            # Ctrl+C belongs to User.stop(), not the disposable transport child.
            start_new_session=True,
        )
    except OSError:
        raise errors("could not start HTTP worker", None, 0, None) from None


def wait(
    output: Future[tuple[bytes, bytes]], cancellation: Event, deadline: float,
) -> Interruption | None:
    for _ in iter(output.done, True):
        if cancellation.is_set():
            return "cancel"
        if (remaining := deadline - monotonic()) <= 0:
            return "timeout"
        _ = cancellation.wait(min(0.02, remaining))
    return None


def exchange(
    request: bytes, cancellation: Event, timeout: float, errors: Failure,
) -> WorkerResult:
    deadline = monotonic() + timeout
    with worker(errors) as process, ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="tai-http-io",
    ) as io:
        # One communicate call owns all stdin writes and drains both output pipes.
        output = io.submit(process.communicate, request)
        try:
            interrupted = wait(output, cancellation, deadline)
        finally:
            if process.poll() is None:
                process.kill()
            # Killing releases blocked writes; collect both streams before leaving.
            _ = process.wait()
        return collect_output(output, process.returncode, interrupted)


def collect_output(
    output: Future[tuple[bytes, bytes]], returncode: int | None,
    interrupted: Interruption | None,
) -> WorkerResult:
    failed = output.exception() is not None
    raw, logs = (b"", b"") if failed else output.result()
    return raw, logs, returncode, interrupted, failed


def mapping(value: object) -> dict[str, object]:
    require(isinstance(value, dict), "expected an object")
    return cast(dict[str, object], value)


def reports(raw: bytes) -> dict[str, object]:
    response: dict[str, object] = {}
    for line in raw.splitlines():
        try:
            record = mapping(json.loads(line))
        except ValueError:
            # A kill can interrupt status writes. Never surface stderr text.
            continue
        response.update(record)
    return response


def failure(result: WorkerResult, response: dict[str, object]) -> str | None:
    _, _, returncode, _, failed = result
    if failed:
        problem: object = "worker communication failed"
    else:
        problem = response.get("error")
    if isinstance(problem, str):
        return problem
    if returncode or not response.get("complete") or response.get("status") is None:
        return "worker failed"
    return None


def check_transport(
    result: WorkerResult, response: dict[str, object], redactor: DiagnosticRedactor,
    errors: Failure,
) -> int:
    reason = failure(result, response)
    raw, _, _, interrupted, _ = result
    status = response.get("status")
    status = status if isinstance(status, int) else None
    fragment = redactor.text(raw.decode("utf-8", errors="replace")) if raw else None
    if interrupted is not None:
        raise errors(interrupted, status, len(raw), fragment)
    if reason == TimeoutError.__name__:
        raise errors("timeout", status, len(raw), fragment)
    if reason is not None:
        raise errors(f"transport failed: {reason}", status, len(raw), fragment)
    assert status is not None
    return status


def check_status(status: int, errors: Failure) -> None:
    if HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
        return
    raise errors("http", status, 0, None)


def content(value: object) -> str:
    body = mapping(value)
    require(ERROR_FIELD not in body, "expected completion choices")
    choices = body["choices"]
    require(isinstance(choices, list), "expected completion choices")
    entries = cast(list[object], choices)
    require(len(entries) == 1, "expected one choice")
    message = mapping(mapping(entries[0])["message"])
    text = message["content"]
    require(message["role"] == ASSISTANT_ROLE and isinstance(text, str), "expected assistant text")
    return cast(str, text)


def completion(result: WorkerResult, redactor: DiagnosticRedactor, errors: Failure) -> str:
    raw, logs, _, _, _ = result
    status = check_transport(result, reports(logs), redactor, errors)
    check_status(status, errors)
    try:
        body: object = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise errors("json", status, len(raw), None) from None
    try:
        return content(body)
    except (KeyError, IndexError, TypeError, ValueError):
        raise errors("content", status, len(raw), None) from None
