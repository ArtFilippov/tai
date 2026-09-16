"""Public execution and observation contracts, independent of any strategy library."""

import json
from dataclasses import dataclass, field
from enum import StrEnum

from tai._diagnostics import DiagnosticObject, DiagnosticRedactor
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
    evaluator_calls: int = 0


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
    _redactor: DiagnosticRedactor = field(
        default_factory=DiagnosticRedactor, repr=False, compare=False
    )

    @property
    def data(self) -> DiagnosticObject:
        """Detached diagnostics; colliding object keys are retained as pairs."""
        return json.loads(self._data_json)

    def dump(self) -> str:
        """Serialize a diagnostic copy, including redaction of envelope fields."""
        return json.dumps(
            self._redactor.snapshot(
                {
                    "run_id": self.run_id,
                    "sequence": self.sequence,
                    "kind": self.kind,
                    "timestamp": self.timestamp,
                    "data": self.data,
                }
            ),
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
