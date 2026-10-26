"""Reusable context for one pinned GGUF Attacker."""

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import override

from target.gguf.gguf_asset import GGUFAsset
from target.gguf.gguf_model_factory import GGUFModelFactory
from target.gguf.gguf_service import GGUFService
from target.gguf.gguf_source import GGUFSource
from target.model import Model


class GGUFRun(GGUFSource):
    """Keep a model usable for the lifetime of its verified service."""

    def __init__(self, file: GGUFAsset, service: GGUFService, tokens: int, factory: GGUFModelFactory) -> None:
        """Keep the service and model source ready for a context-bound run.

        Args:
            file (GGUFAsset): Pinned model file.
            service (GGUFService): Selected service.
            tokens (int): Generation limit.
            factory (GGUFModelFactory): Source of the model.
        """
        super().__init__()
        self.file = file
        self.service = service
        self.tokens = tokens
        self.factory = factory

    @override
    def prepare(self) -> Path:
        return self.file.prepare()

    @override
    def open(self) -> AbstractContextManager[Model]:
        """Yield the model while its verified service is available.

        Returns:
            AbstractContextManager[Model]: Context-bound Attacker.
        """
        @contextmanager
        def session() -> Generator[Model]:
            with self.service.open():
                yield self.factory(
                    self.file, self.service, self.tokens
                )

        return session()
