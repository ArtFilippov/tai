"""Contract for one independent run of a base request."""

from pathlib import Path
from typing import Protocol

from base.base_request import BaseRequest


class RunContract(Protocol):
    """Perform one attempt with a fresh journal at the given path."""

    def run(self, base: BaseRequest, path: Path) -> None: ...
