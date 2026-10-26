"""Public operations on an ordered series of independent ALERT attempts."""

from typing import Any, Protocol
from pathlib import Path

from attack.alert_run import AlertRun
from attack.run_capture_contract import RunCaptureContract
from base.base_request import BaseRequest


class AlertSeriesContract(Protocol):
    """Report recorded attempts and continue the planned series."""

    def report(self) -> dict[str, Any]: ...

    def count(self, category: str, *, key: str = "microcategory") -> dict[str, object]: ...

    def save(self) -> None: ...

    def run(self, runs: AlertRun) -> None: ...

    def invoke(self, runs: AlertRun, base: BaseRequest, path: Path, *, attempt: RunCaptureContract) -> None: ...
