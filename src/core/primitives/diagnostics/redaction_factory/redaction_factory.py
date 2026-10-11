"""Creation of a diagnostic policy from explicit credentials."""

from collections.abc import Sequence
from typing import Protocol

from ..diagnostic_redactor.diagnostic_redactor import DiagnosticRedactor


class RedactionFactory(Protocol):
    """Create an independent policy without coupling callers to its implementation."""

    def __call__(self, credentials: Sequence[str], /) -> DiagnosticRedactor: ...
