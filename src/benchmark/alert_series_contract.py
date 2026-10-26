"""Public operations on an ordered series of independent ALERT attempts."""

from typing import Any, Protocol
from pathlib import Path

from attack.run_contract import RunContract
from attack.run_capture_contract import RunCaptureContract
from base.base_request import BaseRequest


class AlertSeriesContract(Protocol):
    """Report recorded attempts and continue the planned series."""

    def report(self) -> dict[str, Any]: ...

    def count(self, category: str, *, key: str = "microcategory") -> dict[str, object]: ...

    def save(self) -> None: ...

    def run(self, runs: RunContract) -> None: ...

    def invoke(self, runs: RunContract, base: BaseRequest, path: Path, *, attempt: RunCaptureContract) -> None: ...
