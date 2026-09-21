"""Public execution and observation contracts, independent of any strategy library."""

from dataclasses import dataclass, field
from enum import StrEnum

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
    # Admitted evaluations, including errors; model exchanges are journal records.
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
class RunSnapshot:
    """One finished execution's actual state and original, immutable Message."""

    status: UserStatus
    result: Message | None


class User:
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        """Close admission without waiting; the current model/evaluation finishes."""
        raise NotImplementedError

    def status(self) -> UserStatus:
        raise NotImplementedError

    def result(self) -> Message | None:
        raise NotImplementedError

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for worker execution and participant release; never save results."""
        raise NotImplementedError

    def snapshot(self) -> RunSnapshot:
        """Capture a finished worker atomically; reject idle or active execution."""
        raise NotImplementedError
