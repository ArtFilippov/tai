"""An HTTP completion could not be decoded from JSON."""

from typing import override

from .open_ai_response_error import OpenAIResponseError


class OpenAIJSONError(OpenAIResponseError):
    """The response was not valid JSON text."""

    @override
    def category(self) -> str:
        return "json"
