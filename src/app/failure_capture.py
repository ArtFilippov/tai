"""Capture an operation failure while allowing its caller to complete cleanup."""

from types import TracebackType
from typing import Self


class Failure:
    def __init__(self) -> None:
        super().__init__()
        self._error: Exception | None = None

    @property
    def error(self) -> Exception | None:
        return self._error

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, _kind: type[BaseException] | None, value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        if isinstance(value, Exception):
            self._error = value
            return True
        return False

    def reraise(self) -> None:
        if self._error is not None:
            raise self._error
