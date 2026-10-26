"""Journal construction and recovery for independent series attempts."""

from pathlib import Path
from typing import Protocol

from attack.run_capture_contract import RunCaptureContract
from journal.journal import Journal
from journal.journal_view import JournalView


class AlertPorts(Protocol):
    """Create a fresh journal or reader at the chosen attempt path."""

    def journal(self, path: Path) -> Journal: ...

    def view(self, path: Path) -> JournalView: ...

    def capture(self) -> RunCaptureContract: ...
