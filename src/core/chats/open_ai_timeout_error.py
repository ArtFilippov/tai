"""The configured deadline elapsed during an HTTP exchange."""

from typing import override

from .open_ai_transport_error import OpenAITransportError


class OpenAITimeoutError(OpenAITransportError, TimeoutError):
    """The configured whole-exchange deadline expired."""

    @override
    def category(self) -> str:
        return "timeout"
