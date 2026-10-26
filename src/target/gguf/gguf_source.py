"""Reusable preparation and run boundary for a pinned GGUF Attacker."""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from target.model import Model


class GGUFSource(Protocol):
    """Supply a pinned GGUF file and a context-bound Attacker model."""

    def prepare(self) -> Path:
        ...

    def open(self) -> AbstractContextManager[Model]:
        ...
