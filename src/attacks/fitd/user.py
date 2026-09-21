"""A background User running an injected FITD strategy runtime."""

from collections.abc import Callable
from threading import Lock
from typing import override

from core.primitives.messages import Message
from core.runs import RunSnapshot, User, UserStatus

from .execution import Execution
from .execution_state import ExecutionSettings


class FITDUser(User):
    """Independent FITD executions, without diagnostic observation.

    The presented text is the most recent Target response accepted for ladder
    progression in the retained conversation. Context kept for realignment is
    not a result. Use FITDExperiment to assemble this User with a run journal.
    """

    def __init__(self, settings: ExecutionSettings) -> None:
        """Prepare an idle User with explicit execution dependencies.

        Args:
            settings (ExecutionSettings): Participants, task, evaluator and runtime.
        """
        super().__init__()
        self._settings = settings
        self._lock = Lock()
        self._execution = self.new_execution()

    @property
    def base(self) -> Message:
        return self._settings.base

    def new_execution(self, execution_type: Callable[[ExecutionSettings], Execution] = Execution) -> Execution:
        return execution_type(self._settings)

    @override
    def start(self) -> None:
        with self._lock:
            if self._execution.is_active():
                raise RuntimeError("User already has an active run")
            self._start_execution(self.new_execution())

    def _start_execution(self, execution: Execution) -> None:
        _ = execution.start()
        self._execution = execution

    @override
    def stop(self) -> None:
        with self._lock:
            _ = self._execution.stop()

    @override
    def status(self) -> UserStatus:
        with self._lock:
            return self._execution.status()

    @override
    def result(self) -> Message | None:
        with self._lock:
            return self._execution.result()

    @override
    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            execution = self._execution
        return execution.wait(timeout)

    @override
    def snapshot(self) -> RunSnapshot:
        with self._lock:
            return self._execution.snapshot()
