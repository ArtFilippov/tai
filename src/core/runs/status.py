"""Immutable status and result snapshots of an attack run."""

from dataclasses import dataclass, field
from enum import StrEnum

from core.primitives.messages import Message


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
    """One finished execution's actual state and original, immutable Message.

    Attributes:
        status (UserStatus): The actual completed worker status.
        result (Message | None): Original presented result, if one exists.
    """

    status: UserStatus
    result: Message | None
