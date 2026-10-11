"""Per-generation execution inputs and synchronized mutable state."""

from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Literal

from core.chats import LLMChat
from core.evaluation import Evaluator
from core.primitives.messages import Message
from core.runs import UserStatus

from .contracts import StrategyConfiguration, StrategyRuntime

type ModelRole = Literal["Target", "Attacker", "Evaluator"]


@dataclass(frozen=True)
class ExecutionSettings:
    chats: dict[ModelRole, LLMChat]
    base: Message
    evaluator: Evaluator
    configuration: StrategyConfiguration
    runtime: StrategyRuntime


@dataclass
class ExecutionState:
    status: UserStatus
    result: Message | None = None
    worker: Thread | None = None
    stop_requested: Event = field(default_factory=Event)
    closed: bool = False

    def start_worker(self, worker: Thread) -> None:
        self.worker = worker
        worker.start()

    @property
    def working(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    @property
    def stopping(self) -> bool:
        return self.stop_requested.is_set()

    def request_stop(self) -> None:
        self.stop_requested.set()


class RunStopped(Exception):
    """Admission of the next action was refused, not cancellation of a call."""
