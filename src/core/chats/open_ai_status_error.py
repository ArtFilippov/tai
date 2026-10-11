"""A rejected HTTP status outside authentication and rate limiting."""

from typing import override

from .open_ai_http_error import OpenAIHTTPError


class OpenAIStatusError(OpenAIHTTPError):
    """Carry any other non-success HTTP status under the HTTP error contract."""

    @override
    def category(self) -> str:
        return "http"
