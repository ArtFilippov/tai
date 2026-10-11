"""A diagnostic seam for text and detached JSON-shaped evidence."""

from __future__ import annotations

from typing import Protocol


class DiagnosticRedactor(Protocol):
    """Protect text and serialized evidence before it leaves a chat or journal."""

    def secrets(self) -> tuple[str, ...]: ...

    def combine(self, *others: DiagnosticRedactor) -> DiagnosticRedactor: ...

    def text(self, value: str) -> str: ...

    def snapshot(self, value: object, *, mask_fields: bool = False) -> object: ...

    def structured(
        self, value: object, *, mask_fields: bool = False,
    ) -> dict[str, object] | list[object]: ...

    def decode(self, text: str) -> dict[str, object] | list[object]: ...
