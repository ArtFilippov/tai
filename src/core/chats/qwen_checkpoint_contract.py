"""Pinned local model identity without conversation state."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class LocalCheckpointPort(Protocol):
    """Paths and cache identity for one prepared offline model."""

    @property
    def snapshot(self) -> Path: ...

    @property
    def preparation(self) -> str: ...

    def key(self, dtype: str) -> tuple[str, str, str, str]: ...

    def details(self) -> dict[str, object]: ...

    def resolved(self) -> LocalCheckpointPort: ...
