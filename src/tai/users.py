"""Public execution and observation contracts, independent of any strategy library."""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from tai.messages import Message


class UserState(StrEnum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class Progress:
    target_calls: int = 0
    attacker_calls: int = 0
    max_calls_per_conv: int | None = None


@dataclass(frozen=True, slots=True)
class Evaluation:
    result_id: str
    success: bool


@dataclass(frozen=True, slots=True)
class UserStatus:
    state: UserState = UserState.IDLE
    run_id: str | None = None
    progress: Progress = field(default_factory=Progress)
    reason: str | None = None
    result_id: str | None = None
    evaluation: Evaluation | None = None


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence: int
    kind: str
    timestamp: str
    _data_json: str = field(repr=False)

    @property
    def data(self) -> dict[str, Any]:
        """Return a detached value so consumers cannot mutate recorded events."""
        return json.loads(self._data_json)

    def dump(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "sequence": self.sequence,
                "kind": self.kind,
                "timestamp": self.timestamp,
                "data": self.data,
            },
            ensure_ascii=False,
        )


class User:
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        """Request cancellation of active execution without waiting for it."""
        raise NotImplementedError

    def status(self) -> UserStatus:
        raise NotImplementedError

    def result(self) -> Message | None:
        raise NotImplementedError

    def events(self, after: int = 0) -> tuple[RunEvent, ...]:
        raise NotImplementedError

    def wait(self, timeout: float | None = None) -> bool:
        """Optionally block until worker execution and artifact writes finish."""
        raise NotImplementedError
