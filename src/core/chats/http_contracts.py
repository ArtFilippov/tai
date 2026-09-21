"""Immutable observations produced by one disposable HTTP worker."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Interruption(Enum):
    CANCEL = "cancel"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class WorkerResult:
    raw: bytes
    reports: bytes
    returncode: int | None
    interrupted: Interruption | None
    communication_failed: bool

    def text(self, *, errors: Literal["strict", "replace"] = "strict") -> str:
        return self.raw.decode("utf-8", errors=errors)
