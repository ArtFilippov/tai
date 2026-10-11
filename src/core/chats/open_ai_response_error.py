"""Contract for completed HTTP responses without one usable text completion."""

from abc import ABC, abstractmethod
from typing import override

from .open_ai_error import OpenAIError


class OpenAIResponseError(OpenAIError, ABC):
    """A successful HTTP response did not contain a usable text completion."""

    @override
    @abstractmethod
    def category(self) -> str: ...
