"""The strategy-independent lifecycle of a User executing attack runs."""

from core.primitives.messages import Message

from .status import RunSnapshot, UserStatus


class User:
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        """Close admission without waiting; the current model/evaluation finishes."""
        raise NotImplementedError

    def status(self) -> UserStatus:
        raise NotImplementedError

    def result(self) -> Message | None:
        raise NotImplementedError

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for worker execution and participant release; never save results."""
        raise NotImplementedError

    def snapshot(self) -> RunSnapshot:
        """Capture a finished worker atomically; reject idle or active execution."""
        raise NotImplementedError
