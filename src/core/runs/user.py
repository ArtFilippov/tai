"""The strategy-independent lifecycle of a User executing attack runs."""

from core.primitives.messages import Message

from .status import RunSnapshot, UserStatus


class User:
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        """Close admission without waiting; the current model/evaluation finishes.

        Raises:
            NotImplementedError: A concrete User must provide admission control.
        """
        raise NotImplementedError

    def status(self) -> UserStatus:
        raise NotImplementedError

    def result(self) -> Message | None:
        raise NotImplementedError

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for worker execution and participant release; never save results.

        Args:
            timeout (float | None, default=None): Maximum wait in seconds; None waits indefinitely.

        Returns:
            bool: Whether worker execution and cleanup finished.

        Raises:
            NotImplementedError: A concrete User must provide worker waiting.
        """
        raise NotImplementedError

    def snapshot(self) -> RunSnapshot:
        """Capture a finished worker atomically; reject idle or active execution.

        Returns:
            RunSnapshot: Actual worker state and original result.

        Raises:
            NotImplementedError: A concrete User must provide atomic capture.
        """
        raise NotImplementedError
