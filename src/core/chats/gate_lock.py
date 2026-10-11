"""Thread lock actions needed for one exchange or participant reservation."""

from types import TracebackType
from typing import Protocol


class GateLock(Protocol):
    """Lock a critical section without exposing thread implementation types."""

    def __enter__(self) -> bool: ...

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, traceback: TracebackType | None, /,
    ) -> None: ...

    def acquire(self, blocking: bool = True) -> bool: ...

    def release(self) -> None: ...

    def locked(self) -> bool: ...
