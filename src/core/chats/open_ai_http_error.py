"""Contract for rejected Chat Completions HTTP statuses."""

from abc import ABC, abstractmethod
from typing import override

from .open_ai_error import OpenAIError


class OpenAIHTTPError(OpenAIError, ABC):
    """A non-success HTTP status; redirects are not followed."""

    @override
    @abstractmethod
    def category(self) -> str: ...
