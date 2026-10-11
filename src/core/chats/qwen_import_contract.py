"""Lazy optional-library import boundary for local model inference."""

from typing import Protocol


class ImportPort(Protocol):
    """Import a named optional package only during an admitted model call."""

    def __call__(self, name: str, /) -> object: ...
