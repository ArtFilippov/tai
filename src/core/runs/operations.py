"""Internal, immutable outcomes for wrappers around the live FITD operations.

These are domain values, not serialized events. Messages retain their identity;
checkpoint tokens remain opaque and are only interpreted by their LLMChat.
"""

from dataclasses import dataclass, field
from typing import Literal

from core.chats import ModelRole
from core.primitives.messages import Message

from .status import Evaluation, UserStatus


@dataclass(frozen=True, slots=True)
class Cause:
    reason: str
    attempt_id: str | None = None
    id: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class Context:
    run_id: str
    role: ModelRole
    conversation_id: str
    cause: Cause


@dataclass(frozen=True, slots=True)
class Exchange:
    context: Context
    call_id: str
    request: Message
    response: Message


@dataclass(frozen=True, slots=True)
class Checkpoint:
    context: Context
    id: str
    token: object = field(repr=False)


@dataclass(frozen=True, slots=True)
class ConversationChange:
    kind: Literal["unchanged", "restored", "reset"]
    context: Context
    previous_conversation_id: str
    checkpoint: Checkpoint | None = None


@dataclass(frozen=True, slots=True)
class ResultChange:
    kind: Literal["accepted", "replaced", "withdrawn", "unchanged"]
    context: Context
    checkpoint_id: str
    previous: Message | None
    current: Message | None
    previous_evaluation: Evaluation | None
    evaluation: Evaluation | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    context: Context
    checkpoint_id: str
    result: Message
    evaluation: Evaluation
    base: Message


@dataclass(frozen=True, slots=True)
class ConversationPoint:
    checkpoint: Checkpoint
    request: Message | None
    response: Message | None
    result: Message | None


@dataclass(frozen=True, slots=True)
class Synchronization:
    conversation: ConversationChange
    before: ConversationPoint
    after: ConversationPoint
    result: ResultChange


@dataclass(frozen=True, slots=True)
class TargetExchange:
    exchange: Exchange
    synchronization: Synchronization
    point: ConversationPoint


@dataclass(frozen=True, slots=True)
class LocalDecision:
    kind: Literal["refusal", "alignment"]
    value: bool
    context: Context
    request: Message
    response: Message
    checkpoint_id: str | None
    judgment: Exchange | None = None


@dataclass(frozen=True, slots=True)
class BranchSelection:
    id: str
    context: Context
    branch: Literal["exhausted", "continue", "realign", "restart", "rephrase"]
    point: ConversationPoint


@dataclass(frozen=True, slots=True)
class BranchDecision(BranchSelection):
    next_attempt_id: str | None
    next_request: Message | None
    levels_completed: int
    next_levels_completed: int | None
    checks: tuple[LocalDecision, ...]


@dataclass(frozen=True, slots=True)
class RunStarted:
    """Admission only; the worker may still be executing when this is returned.

    Attributes:
        run_id (str): Identity of the admitted execution.
    """

    run_id: str


@dataclass(frozen=True, slots=True)
class ResourcesReleased:
    run_id: str
    roles: tuple[ModelRole, ...]


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    """Returned by worker execution, including finalization and resource release.

    Attributes:
        status (UserStatus): Terminal worker status.
        result (Message | None): Original presented result, if available.
        error (Exception | None): Original execution or cleanup failure.
        released (ResourcesReleased | None): Completed resource-release evidence.
    """

    status: UserStatus
    result: Message | None
    error: Exception | None
    released: ResourcesReleased | None
