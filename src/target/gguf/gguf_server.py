"""A selected GGUF service with explicit process ownership."""

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from typing import override

from target.gguf.gguf_address import GGUFAddress
from target.gguf.gguf_asset import GGUFAsset
from target.gguf.gguf_service import GGUFService
from target.gguf.gguf_service_factory import GGUFServiceFactory


class GGUFServer(GGUFService):
    """Bind a pinned file to a verified server address and CUDA device."""

    def __init__(self, file: GGUFAsset, location: GGUFAddress, gpu: int, factory: GGUFServiceFactory) -> None:
        """Keep the selected settings without launching the server.

        Args:
            file (GGUFAsset): Prepared weights and required model identity.
            location (GGUFAddress): Selected host and port.
            gpu (int): Selected CUDA device.
            factory (GGUFServiceFactory): Lifecycle implementation.
        """
        super().__init__()
        self.file = file
        self.location = location
        self.gpu = gpu
        self.factory = factory
        self.ownership: list[bool] = []

    @override
    def open(self) -> AbstractContextManager[bool]:
        """Keep one verified service available for the duration of a context.

        Returns:
            AbstractContextManager[bool]: Server ownership within the context.
        """
        @contextmanager
        def session() -> Generator[bool]:
            with self.factory(
                self.location.host(),
                self.location.port(),
                self.file.local(),
                self.gpu,
                expected=self.file.identity()
            ) as owned:
                self.ownership.append(owned)
                try:
                    yield owned
                finally:
                    _ = self.ownership.pop()

        return session()

    @override
    def address(self) -> str:
        return self.location.origin()

    @override
    def settings(self) -> dict[str, object]:
        owned = self.ownership[-1]
        return {
            "server": "owned" if owned else "reused",
            "thinking": False if owned else "server-controlled",
            "device": f"cuda:{self.gpu}" if owned else "server-controlled",
            "server_settings": {
                "n_gpu_layers": -1,
                "main_gpu": 0,
                "n_ctx": 2048,
                "enable_thinking": False
            } if owned else "server-controlled"
        }
