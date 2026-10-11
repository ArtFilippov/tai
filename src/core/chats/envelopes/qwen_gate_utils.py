"""Cancellable admission to the shared local generation gate."""

from collections.abc import Generator
from contextlib import contextmanager
from threading import Event, Lock

from .cancelled import Cancelled


def check_cancelled(cancellation: Event) -> None:
    if cancellation.is_set():
        raise Cancelled()


@contextmanager
def admitted(gate: Lock, cancellation: Event) -> Generator[None]:
    check_cancelled(cancellation)
    for _ in iter(lambda: gate.acquire(timeout=0.02), True):
        check_cancelled(cancellation)
    try:
        yield
    finally:
        gate.release()
