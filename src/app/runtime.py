"""System clock and main-thread interrupt ownership."""

import signal
from collections.abc import Callable
from datetime import UTC, datetime
from types import FrameType, TracebackType
from typing import Self, override

from .contracts import StopRequests


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Interrupts(StopRequests):
    def __init__(self) -> None:
        super().__init__()
        self._interrupted = False
        self._accepting = True
        self._previous: Callable[[int, FrameType | None], object] | int | None = signal.SIG_DFL

    @property
    @override
    def interrupted(self) -> bool:
        return self._interrupted

    def __enter__(self) -> Self:
        self._previous = signal.signal(signal.SIGINT, self._receive_signal)
        return self

    def __exit__(
        self, _kind: type[BaseException] | None, _value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        _ = signal.signal(signal.SIGINT, self._previous)

    def _receive_signal(self, _signum: int, _frame: FrameType | None) -> None:
        self.request()

    @override
    def request(self) -> None:
        # Never raise through join, take a User lock, or perform I/O here.
        if self._accepting:
            self._interrupted = True

    @override
    def seal(self) -> None:
        self._accepting = False
