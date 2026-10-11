"""Device operations required by the optional Torch runtime."""

from contextlib import AbstractContextManager
from typing import Protocol


class Cuda(Protocol):
    """Bound local GPU and synchronization for one completed request."""

    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def is_bf16_supported(self) -> bool: ...

    def device(self, device: str) -> AbstractContextManager[object]: ...

    def synchronize(self, device: str) -> None: ...
