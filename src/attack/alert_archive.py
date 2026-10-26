"""Boundary for persisting and recovering an ALERT series."""

from pathlib import Path
from typing import Any, Protocol

from attack.run_capture_contract import RunCaptureContract


class AlertArchive(Protocol):
    """Commit a series index and classify only saved run outcomes."""

    def save(self, path: Path, report: dict[str, Any]) -> None: ...

    def record(self, path: Path, event: dict[str, object]) -> None: ...

    def failure(self, path: Path) -> str: ...

    def outcome(self, path: Path) -> dict[str, str]: ...

    def categories(self) -> list[str]: ...

    def capture(self) -> RunCaptureContract: ...
