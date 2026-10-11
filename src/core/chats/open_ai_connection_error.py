"""A non-timeout failure while exchanging a Chat Completions request."""

from typing import override

from .open_ai_transport_error import OpenAITransportError


class OpenAIConnectionError(OpenAITransportError):
    """A disposable HTTP worker or connection failed."""

    @override
    def category(self) -> str:
        return "transport"
