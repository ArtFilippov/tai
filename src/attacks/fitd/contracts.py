"""Strategy ports: native library objects never cross the execution boundary."""

from abc import abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from attacks.shared.contracts import CompletionReason
from core.primitives.messages import Message
from core.runs.operations import (
    BranchDecision, Cause, Context, ConversationPoint,
    LocalDecision, ResultChange, Synchronization, TargetExchange,
)

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
