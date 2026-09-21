"""Strategy ports: native library objects never cross the execution boundary."""

from abc import abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

from core.chats import ModelRole
from core.primitives.messages import Message
from core.runs.operations import (
    BranchDecision, Cause, Checkpoint, Context, ConversationChange, ConversationPoint,
    EvaluationResult, Exchange, LocalDecision, ResultChange, Synchronization, TargetExchange,
)

from .profile import FITDProfile

type CompletionReason = Literal["evaluator_success", "round_limit", "strategy_exhausted"]


@dataclass(frozen=True)
class ConversationPorts:
    checkpoint: Callable[[ModelRole, Cause], Checkpoint]
    restore: Callable[[ModelRole, Checkpoint, Cause], ConversationChange]
    reset: Callable[[ModelRole, Cause], ConversationChange]
    conversation_id: Callable[[ModelRole], str]


@dataclass(frozen=True)
class StrategyPorts:
    send: Callable[[ModelRole, Message, Cause], Exchange]
    publish: Callable[[Message | None, Context, str], ResultChange]
    evaluate: Callable[[Message, Context, str], EvaluationResult]
    check_running: Callable[[], None]
    conversation: ConversationPorts


class StrategyRuntime(Protocol):
    @property
    @abstractmethod
    def revision(self) -> str: ...

    @abstractmethod
    def __call__(self, base: Message, ports: StrategyPorts, profile: FITDProfile) -> CompletionReason: ...


@dataclass(frozen=True)
class Continuation[Attempt]:
    decision: BranchDecision
    attempt: Attempt | None


@dataclass(frozen=True)
class Selection:
    cause: Cause
    point: ConversationPoint
    checks: tuple[LocalDecision, ...]


class ProbeOperations[Prompt, Attempt](Protocol):
    """Execute a configured probe and observe operations over opaque native inputs."""

    @abstractmethod
    def __call__(self) -> CompletionReason: ...

    @abstractmethod
    def raise_if_interrupted(self) -> None: ...

    @abstractmethod
    def check_refusal(
        self, request: Message, response: Message, context: Context, checkpoint_id: str | None,
    ) -> LocalDecision: ...

    @abstractmethod
    def check_alignment(
        self, request: Message, response: Message, context: Context, checkpoint_id: str,
    ) -> LocalDecision: ...

    @abstractmethod
    def select_continuation(self, conversation: Prompt, last_attempt: Attempt) -> Continuation[Attempt]: ...

    @abstractmethod
    def generate_next_turn_attempt(self, conversation: Prompt, last_attempt: Attempt) -> Attempt | None: ...

    @abstractmethod
    def completion_reason(self) -> CompletionReason: ...


class TargetOperations[Prompt, Response](Protocol):
    """Native conversation synchronization exposing only domain outcomes."""

    @property
    @abstractmethod
    def current(self) -> ConversationPoint: ...

    @abstractmethod
    def attempt(self, attempt_id: str, request: Message) -> AbstractContextManager[None]: ...

    @abstractmethod
    def save(
        self, request: Message | None, response: Message | None, result: Message | None, cause: Cause,
    ) -> ConversationPoint: ...

    @abstractmethod
    def synchronize(self, prompt: Prompt, cause: Cause) -> Synchronization: ...

    @abstractmethod
    def accept(self, cause: Cause) -> ResultChange: ...

    @abstractmethod
    def exchange(self, prompt: Prompt, request: Message, cause: Cause) -> TargetExchange: ...

    @abstractmethod
    def generate(self, prompt: Prompt, generations_this_call: int) -> list[Response]: ...

    @abstractmethod
    def clear_history(self) -> None: ...
