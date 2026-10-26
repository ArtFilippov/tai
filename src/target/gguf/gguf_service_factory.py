"""Contract for starting or reusing a GGUF service."""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol


class GGUFServiceFactory(Protocol):
    """Create a context for a service with an exact model identity."""

    def __call__(self, host: str, port: int, path: Path, gpu: int, *, expected: str) -> AbstractContextManager[bool]:
        ...
