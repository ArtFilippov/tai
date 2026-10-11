"""Contract for connection, framing and worker failures."""

from abc import ABC, abstractmethod
from typing import override

from .open_ai_error import OpenAIError


class OpenAITransportError(OpenAIError, ABC):
    """Connection, TLS, HTTP framing, or worker failure."""

    @override
    @abstractmethod
    def category(self) -> str: ...
