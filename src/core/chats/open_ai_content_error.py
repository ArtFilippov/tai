"""A successful JSON response lacked one assistant text completion."""

from typing import override

from .open_ai_response_error import OpenAIResponseError


class OpenAIContentError(OpenAIResponseError):
    """The response did not contain exactly one assistant text completion."""

    @override
    def category(self) -> str:
        return "content"
