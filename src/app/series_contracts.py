"""Values and ports for sequential, independently persisted attack runs."""

from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from alert.accounting import SeriesReport
from core.primitives.messages import Message
from core.runs import RunSnapshot

from .contracts import Clock, RunHandle, StopRequests


class SeriesStore(Protocol):
    @property
    def directory(self) -> Path: ...

    @abstractmethod
    def initialize(self) -> dict[str, object]: ...

    def persist(self, index: Mapping[str, object], report: SeriesReport) -> None: ...

    def inspect(self, snapshot: RunSnapshot | None) -> dict[str, object]: ...

    def finish(self) -> None: ...


@dataclass(frozen=True)
class SeriesDependencies:
    create: Callable[[Message], RunHandle]
    store: SeriesStore
    stop: StopRequests
    clock: Clock
    redact: Callable[[str], str]


@dataclass(frozen=True)
class SeriesResult:
    directory: Path
    report: SeriesReport
    exit_code: int


def snapshot_outcome(snapshot: RunSnapshot | None) -> dict[str, object]:
    status = snapshot.status if snapshot is not None else None
    result = snapshot.result if snapshot is not None else None
    return {
        "state": status.state.value if status is not None else None,
        "reason": status.reason if status is not None else None,
        "run_id": status.run_id if status is not None else None,
        "progress": asdict(status.progress) if status is not None else None,
        "result_id": status.result_id if status is not None else None,
        "result": {"kind": "Message", "id": result.id} if isinstance(result, Message) else None,
        "evaluation": asdict(status.evaluation) if status is not None and status.evaluation is not None else None,
    }
