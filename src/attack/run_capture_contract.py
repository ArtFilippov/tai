"""Boundary for preserving an attempt failure while continuing a series."""

from contextlib import AbstractContextManager
from typing import Protocol


class RunCaptureContract(Protocol):
    """Capture ordinary exceptions and propagate interruptions."""

    def guard(self) -> AbstractContextManager[object]: ...

    def failed(self) -> bool: ...

    def error(self) -> str: ...
