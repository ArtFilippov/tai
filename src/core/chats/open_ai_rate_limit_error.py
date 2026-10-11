"""A non-retried HTTP request exceeded the provider's limit."""

from typing import override

from .open_ai_http_error import OpenAIHTTPError


class OpenAIRateLimitError(OpenAIHTTPError):
    """The endpoint limited the request (429); no implicit retry is made."""

    @override
    def category(self) -> str:
        return "rate_limit"
