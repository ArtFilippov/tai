"""HTTP credentials or authorization were rejected."""

from typing import override

from .open_ai_http_error import OpenAIHTTPError


class OpenAIAuthenticationError(OpenAIHTTPError):
    """The endpoint rejected authentication or access (401/403)."""

    @override
    def category(self) -> str:
        return "authentication"
