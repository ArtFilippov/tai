"""Capture a failed attempt without stopping the remaining series."""

from contextlib import AbstractContextManager
from types import TracebackType
from typing import Self, override

from attack.run_capture_contract import RunCaptureContract


class RunCapture(RunCaptureContract):
    """Retain an ordinary run exception while propagating interruptions."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    @override
    def guard(self) -> AbstractContextManager[object]:
        """Expose the exception boundary for one invocation.

        Returns:
            AbstractContextManager[object]: This attempt's context.
        """
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None,
        trace: TracebackType | None,
    ) -> bool:
        if isinstance(
            error, Exception
        ):
            self.errors.append(
                repr(error)
            )
            return True
        return False

    @override
    def failed(self) -> bool:
        """Report whether an ordinary run failure was captured.

        Returns:
            bool: Whether the attempt failed.
        """
        return bool(self.errors)

    @override
    def error(self) -> str:
        """Return the captured failure after a failed attempt.

        Returns:
            str: Recorded run failure.

        Raises:
            ValueError: If this attempt did not fail.
        """
        if not self.errors:
            raise ValueError("Attempt has no captured failure")
        return self.errors[0]
