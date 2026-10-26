"""Contract for a pinned local GGUF quantization."""

from pathlib import Path
from typing import Protocol


class GGUFAsset(Protocol):
    """Prepare and identify a single pinned GGUF file."""

    def prepare(self) -> Path:
        ...

    def local(self) -> Path:
        ...

    def identity(self) -> str:
        ...

    def commit(self) -> str:
        ...

    def details(self) -> dict[str, object]:
        ...
