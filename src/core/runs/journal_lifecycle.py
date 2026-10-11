"""Journal lifecycle reads, failure propagation, and configuration snapshots."""

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import override

from core.primitives.diagnostics import DiagnosticRedactor

from .journal_configuration import JournalConfiguration, JournalOptions
from .journal_output import JournalOutput
from .journal_ports import JournalLifecyclePort
from .journal_state import JournalState
from .journal_support import JournalOutputPort, JournalStatePort
from .journal_values import operation_value
from .status import Progress, UserStatus


__all__ = ["JournalConfiguration", "JournalLifecycle", "JournalOptions"]


@dataclass
class _ConfigurationSnapshots:
    initial: object = None
    final: object = None


class JournalLifecycle(JournalLifecyclePort):
    def __init__(
        self, *, redactor: DiagnosticRedactor,
        max_calls_per_conv: int | None = None,
        settings: Callable[[], Mapping[str, object]] = dict,
        final_settings: Callable[[], Mapping[str, object]] = dict,
    ) -> None:
        super().__init__()
        self.redactor: DiagnosticRedactor = redactor
        self._configuration = JournalConfiguration(settings, final_settings)
        self._snapshots = _ConfigurationSnapshots()
        self._output: JournalOutputPort = JournalOutput()
        self._state: JournalStatePort = JournalState(UserStatus(
            progress=Progress(max_calls_per_conv=max_calls_per_conv)
        ))

    @property
    @override
    def error(self) -> Exception | None:
        """Return the original failure, even when diagnostic storage failed.

        Returns:
            Exception | None: First diagnostic/storage failure, if any.
        """
        with self._state.lock:
            return self._state.error

    @property
    @override
    def finished(self) -> bool:
        return self._state.finished()

    @override
    def wait(self, timeout: float | None = None) -> bool:
        """Wait for actual work and journal closure, independently of User.wait.

        Args:
            timeout (float | None, default=None): Maximum wait in seconds; None waits indefinitely.

        Returns:
            bool: Whether the journal finished before the timeout.
        """
        return self._state.wait(timeout)

    @override
    def status(self) -> UserStatus:
        with self._state.lock:
            return self._state.status

    @override
    def check(self) -> None:
        if (error := self.error) is not None:
            raise error
        if self.finished:
            raise RuntimeError("Journal is closed")

    @contextmanager
    def _guard(self) -> Generator[None]:
        try:
            yield
        except Exception as exc:
            self._state.poison(exc, self.redactor)
            raise

    @override
    def settings(self) -> object:
        """Return detached settings observed at initialization, if available.

        Returns:
            object: Detached initial configuration, or None before capture.
        """
        with self._state.lock:
            return deepcopy(self._snapshots.initial)

    @override
    def final_settings(self) -> object:
        """Return participant settings captured before ownership release.

        Returns:
            object: Detached final configuration, or None before capture.
        """
        with self._state.lock:
            return deepcopy(self._snapshots.final)

    def _capture_initial_settings(self) -> object:
        snapshot = self.redactor.snapshot(operation_value(self._configuration.source()))
        with self._state.lock:
            self._snapshots.initial = snapshot
        return snapshot

    @override
    def capture_final_settings(self) -> None:
        self.check()
        with self._guard():
            settings = self.redactor.snapshot(operation_value(self._configuration.final_source()))
            with self._state.lock:
                self._snapshots.final = settings
