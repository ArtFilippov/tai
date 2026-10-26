"""Forward GGUF preparation and opening to an injected source."""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import override

from target.gguf_source import GGUFSource
from target.model import Model


class GGUFSourceEnvelope(GGUFSource):
    """Preserve the GGUF source contract for composed profiles."""

    def __init__(self, delegate: GGUFSource) -> None:
        """Hold one source for every operation.

        Args:
            delegate (GGUFSource): Source of the selected model.
        """
        self.delegate = delegate

    @override
    def prepare(self) -> Path:
        return self.delegate.prepare()

    @override
    def open(self) -> AbstractContextManager[Model]:
        return self.delegate.open()
