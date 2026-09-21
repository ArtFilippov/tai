"""Conversation snapshots and the nonblocking exchange gate."""

from dataclasses import dataclass
from contextlib import ExitStack
from threading import Event, Lock
from types import TracebackType
from typing import Literal, Self
from uuid import uuid4

from core.primitives.messages import Message


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: Literal["system", "user", "assistant"]
    message: Message

    def text(self) -> str:
        return self.message.to_string()

    def dump(self) -> str:
        return self.message.dump()


class ConversationState:
    def __init__(self, turns: tuple[ChatTurn, ...], system: tuple[ChatTurn, ...]) -> None:
        super().__init__()
        self.turns = turns
        self.system = system
        self.identity = str(uuid4())
        self.checkpoints: dict[object, tuple[ChatTurn, ...]] = {}

    def reset(self) -> None:
        self.identity = str(uuid4())
        self.turns = self.system
        self.checkpoints.clear()

    def restore(self, token: object) -> None:
        # Identity comparison also rejects unhashable inputs without consulting
        # caller-defined equality/hash methods on an opaque checkpoint boundary.
        for known, turns in self.checkpoints.items():
            if known is token:
                self.turns = turns
                return
        raise ValueError("Foreign or stale LLMChat checkpoint")


class ExchangeGate:
    def __init__(self) -> None:
        super().__init__()
        self.lock = Lock()
        self.sending = Lock()
        self.active: Event | None = None
        self.owner: object | None = None

    def __enter__(self) -> Self:
        if not self.sending.acquire(blocking=False):
            raise RuntimeError("LLMChat already has an active exchange")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with self.lock:
            self.active = None
        self.sending.release()

    def busy(self) -> bool:
        return self.owner is not None or self.sending.locked()

    def cancel(self) -> None:
        with self.lock:
            if self.active is not None:
                self.active.set()

    @staticmethod
    def acquire_all(gates: tuple["ExchangeGate", ...], owner: object) -> None:
        with ExitStack() as locks:
            for gate in gates:
                _ = locks.enter_context(gate.lock)
            if any(gate.busy() for gate in gates):
                raise RuntimeError("LLMChat already belongs to an active execution")
            for gate in gates:
                gate.owner = owner

    @staticmethod
    def release_all(gates: tuple["ExchangeGate", ...], owner: object) -> None:
        with ExitStack() as locks:
            for gate in gates:
                _ = locks.enter_context(gate.lock)
            for gate in gates:
                if gate.owner is owner:
                    gate.owner = None
