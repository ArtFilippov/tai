"""Snapshot and redaction contract shared by conversation roles."""

from abc import abstractmethod
from typing import Protocol

from core.primitives.diagnostics import DiagnosticRedactor


class ChatSettings(Protocol):
    """Keep transport-specific settings behind a detached diagnostic snapshot."""

    @property
    def redactor(self) -> DiagnosticRedactor: ...

    @abstractmethod
    def settings_data(self) -> dict[str, object]: ...

    def settings(self) -> dict[str, object] | list[object]:
        """Return the detached diagnostic snapshot.

        Returns:
            dict[str, object] | list[object]: Safe snapshot.
        """
        return self.redactor.structured(self.settings_data())

    def diagnostic_settings(
        self, redactor: DiagnosticRedactor,
    ) -> dict[str, object] | list[object]:
        """Apply a run-wide redaction policy to the snapshot.

        Args:
            redactor (DiagnosticRedactor): Policy shared across roles.

        Returns:
            dict[str, object] | list[object]: Safe snapshot.
        """
        return redactor.structured(self.settings())
