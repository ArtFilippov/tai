"""Per-generation execution inputs and synchronized mutable state."""

from dataclasses import dataclass, field
from threading import Event, Thread

from core.chats import LLMChat, ModelRole
from core.evaluation import Evaluator
from core.primitives.messages import Message
from core.runs import UserStatus

from .contracts import StrategyRuntime
from .profile import FITDProfile


@dataclass(frozen=True)
class ExecutionSettings:
    chats: dict[ModelRole, LLMChat]
    base: Message
    evaluator: Evaluator
    profile: FITDProfile
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
