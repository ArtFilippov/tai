"""A free-form base request without source taxonomy."""

from collections.abc import Mapping
from typing import override

from base.base_request import BaseRequest


class PlainBase(BaseRequest):
    """An original text request without a pinned external origin."""

    def __init__(self, prompt: str) -> None:
        """Store the plain text.

        Args:
            prompt (str): Original request.
        """
        super().__init__()
        self.prompt = prompt

    @override
    def text(self) -> str:
        return self.prompt

    @override
    def metadata(self) -> Mapping[str, str | int]:
        return {}
