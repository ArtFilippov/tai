"""Observation of real transport processes and their pipe lifetimes."""

import os
import signal
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event, Thread, current_thread
from typing import TypedDict, Unpack

import pytest

from tai import OpenAIChat


class WorkerOptions(TypedDict, total=False):
    stdin: int
    stdout: int
    stderr: int
    start_new_session: bool
    pipesize: int


def _observed_call[**P, R](
    call: Callable[P, R], observe: Callable[[], None],
) -> Callable[P, R]:
    def invoke(*args: P.args, **kwargs: P.kwargs) -> R:
        observe()
        return call(*args, **kwargs)
    return invoke


@dataclass
class Workers:
    processes: list[subprocess.Popen[bytes]] = field(default_factory=list[subprocess.Popen[bytes]])
    communicators: list[Thread] = field(default_factory=list[Thread])

    def assert_reaped(self, *, pipes: bool = False) -> None:
        for worker in self.processes:
            assert worker.returncode is not None
            if pipes:
                assert worker.stdin is not None and worker.stdin.closed
                assert worker.stdout is not None and worker.stdout.closed
                assert worker.stderr is not None and worker.stderr.closed
            with pytest.raises(ChildProcessError):
                os.waitpid(worker.pid, os.WNOHANG)
        assert all(not thread.is_alive() for thread in self.communicators)

    def resume(self) -> None:
        for worker in self.processes:
            if worker.poll() is None:
                worker.send_signal(signal.SIGCONT)

    def cancel(self, chat: OpenAIChat) -> None:
        chat.cancel()
        self.resume()


def gate_worker_input(monkeypatch: pytest.MonkeyPatch) -> tuple[Event, Workers]:
    """Stop a real child before stdin reads, with a known 4096-byte pipe capacity.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped patch owner.

    Returns:
        tuple[Event, Workers]: Pipe-filled signal and observed process/thread lifetimes.
    """
    workers = Workers()
    filled = Event()
    written: dict[int, int] = {}
    write = os.write

    class GatedWorker(subprocess.Popen[bytes]):
        def __init__(self, args: list[str], **kwargs: Unpack[WorkerOptions]) -> None:
            kwargs["pipesize"] = 4096
            super().__init__(args, **kwargs)
            self.send_signal(signal.SIGSTOP)
            assert self.stdin is not None
            written[self.stdin.fileno()] = 0
            workers.processes.append(self)

    def observe_write(fd: int, data: bytes) -> int:
        count = write(fd, data)
        if fd in written:
            written[fd] += count
            if written[fd] >= 4096:
                filled.set()
        return count

    monkeypatch.setattr(subprocess, "Popen", GatedWorker)
    monkeypatch.setattr(GatedWorker, "communicate", _observed_call(
        GatedWorker.communicate, lambda: workers.communicators.append(current_thread()),
    ))
    monkeypatch.setattr(os, "write", observe_write)
    return filled, workers


def failing_worker(monkeypatch: pytest.MonkeyPatch) -> Workers:
    workers = Workers()

    class FailedCommunication(subprocess.Popen[bytes]):
        def __init__(self, args: list[str], **kwargs: Unpack[WorkerOptions]) -> None:
            super().__init__(args, **kwargs)
            workers.processes.append(self)

    def fail() -> None:
        workers.communicators.append(current_thread())
        raise OSError("synthetic pipe failure")

    monkeypatch.setattr(subprocess, "Popen", FailedCommunication)
    monkeypatch.setattr(FailedCommunication, "communicate", _observed_call(
        FailedCommunication.communicate, fail,
    ))
    return workers


def observe_workers(monkeypatch: pytest.MonkeyPatch) -> Workers:
    workers = Workers()

    class ObservedWorker(subprocess.Popen[bytes]):
        def __init__(self, args: list[str], **kwargs: Unpack[WorkerOptions]) -> None:
            super().__init__(args, **kwargs)
            workers.processes.append(self)

    monkeypatch.setattr(subprocess, "Popen", ObservedWorker)
    return workers


def observe_partial_worker(monkeypatch: pytest.MonkeyPatch, prefix: bytes) -> tuple[Event, Workers]:
    """Synchronize cancellation with actual pipe delivery, not a server-side write.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped patch owner.
        prefix (bytes): Response bytes that must have reached the parent.

    Returns:
        tuple[Event, Workers]: Delivery signal and observed processes.
    """
    observed = Event()
    workers = Workers()
    read = os.read
    received: dict[int, bytearray] = {}

    class ObservedWorker(subprocess.Popen[bytes]):
        def __init__(self, args: list[str], **kwargs: Unpack[WorkerOptions]) -> None:
            super().__init__(args, **kwargs)
            workers.processes.append(self)
            assert self.stdout is not None
            received[self.stdout.fileno()] = bytearray()

    def observe_read(fd: int, size: int) -> bytes:
        chunk = read(fd, size)
        if fd in received:
            received[fd].extend(chunk)
            if prefix in received[fd]:
                observed.set()
        return chunk

    monkeypatch.setattr(subprocess, "Popen", ObservedWorker)
    monkeypatch.setattr(os, "read", observe_read)
    return observed, workers
