"""Contract for native deletion variants with provenance."""

from collections.abc import Iterator, Mapping
from typing import Protocol

class DeletionVariants(Protocol):
    """Yield the native transformed request and operation details."""

    def generate(self, base: str) -> Iterator[Mapping[str, object]]:
        ...
