"""Contract for independent attempts within a base-request series."""

from pathlib import Path
from typing import Protocol

from base.base_request import BaseRequest


class AlertRun(Protocol):
    """Perform one attempt with a fresh journal at the given path."""

    def run(self, base: BaseRequest, path: Path) -> None: ...
