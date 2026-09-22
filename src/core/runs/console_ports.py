"""Detached diagnostic updates delivered directly to an addressed console row."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DisplayExchange:
    role: str
    call_id: str
    request: str
    response: str | None = None


@dataclass(frozen=True)
class RowChange:
    state: str | None = None
    method: str | None = None
    run_id: str | None = None
    exchange: DisplayExchange | None = None
    finished: bool | None = None


class ConsoleRow(Protocol):
    def update(self, change: RowChange) -> None: ...
