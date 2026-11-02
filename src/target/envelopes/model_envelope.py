"""Forward model operations to an injected delegate."""

from collections.abc import Sequence
from typing import override

from target.model import Model


class ModelEnvelope(Model):
    """Preserve the Model interface for composed model variants."""

    def __init__(self, delegate: Model) -> None:
        """Store one model delegate.

        Args:
            delegate (Model): Model receiving every operation.
        """
        self.delegate = delegate

    @override
    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        return self.delegate.generate(messages)

    @override
    def prefill(self, messages: Sequence[dict[str, str]], prefix: str) -> str:
        return self.delegate.prefill(
            messages, prefix
        )

    @override
    def revision(self) -> str:
        return self.delegate.revision()

    @override
    def settings(self) -> dict[str, object]:
        return self.delegate.settings()
