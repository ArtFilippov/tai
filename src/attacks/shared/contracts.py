"""Strategy ports and parameters, independent of native library types."""

from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from core.chats import ModelRole
from core.primitives.messages import Message
from core.runs.operations import Cause, Checkpoint, Context, ConversationChange, EvaluationResult, Exchange, ResultChange

GARAK_REVISION = "3f50ea5ff9cd7050099940647c15c39b07a93392"
type CompletionReason = Literal["evaluator_success", "round_limit", "strategy_exhausted"]
type AttackType = Literal["fitd", "adaptive_attacks", "badchars_deletion", "dra"]
FITD: Final[AttackType] = "fitd"
ADAPTIVE_ATTACKS: Final[AttackType] = "adaptive_attacks"
BADCHARS_DELETION: Final[AttackType] = "badchars_deletion"
DRA: Final[AttackType] = "dra"


@dataclass(frozen=True)
class Candidate:
    request: Message
    cause: Cause


@dataclass(frozen=True)
class StrategyConfiguration:
    attack_type: AttackType
    parameters: Mapping[str, object]
    max_calls_per_conv: int | None = None

    def settings(self) -> dict[str, object]:
        return {"attack": {"type": self.attack_type}, **self.parameters}


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
    def __call__(self, base: Message, ports: StrategyPorts) -> CompletionReason: ...
