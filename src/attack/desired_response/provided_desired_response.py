"""A supplied answer bound to an exact base request."""

from typing import override

from attack.desired_response.desired_response import DesiredResponse
from base.base_request import BaseRequest
from journal.journal import Journal


class ProvidedDesiredResponse(DesiredResponse):
    """Keep one supplied answer associated with its original request."""

    def __init__(self, base: BaseRequest, text: str) -> None:
        """Bind supplied text to one exact request.

        Args:
            base (BaseRequest): Original request with provenance.
            text (str): Prepared answer.
        """
        super().__init__()
        self.base = base
        self.text = text

    @override
    def response(self, base: str, journal: Journal) -> str:
        if base != self.base.text() or not self.text.strip():
            raise ValueError("Missing desired response for this base request")
        return self.text

    @override
    def description(self) -> dict[str, object]:
        return {"type": "provided_desired_response"}
