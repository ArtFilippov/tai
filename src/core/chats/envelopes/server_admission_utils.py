"""Release each admitted request even when transport or decoding fails."""

from collections.abc import Generator
from contextlib import contextmanager

from ..server_lifecycle import ServerLifecycle
from .qwen_server_error import QwenServerError


@contextmanager
def admitted(lifecycle: ServerLifecycle) -> Generator[None]:
    if not lifecycle.admit():
        raise QwenServerError("vLLM server is closed")
    try:
        yield
    finally:
        lifecycle.release()
