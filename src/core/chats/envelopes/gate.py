"""Exclusive exchange and execution reservation for one conversation."""

from threading import Event, Lock
from types import TracebackType
from typing import Self, override

from core.primitives.messages import Message

from ..exchange_gate_contract import ExchangeGate
from ..gate_lock import GateLock
from .cancelled import Cancelled


class Gate(ExchangeGate):
    """Serialize sends, coordinate cancellation and reserve one participant.

    Attributes:
        lock (GateLock): Protect exchange signals and participant reservations.
        sending (GateLock): Deny concurrent exchanges on this conversation.
    """

    lock: GateLock
    sending: GateLock

    def __init__(self) -> None:
        super().__init__()
        self.lock, self.sending = Lock(), Lock()
        self.active: list[Event] = []
        self.owner: list[object] = []

    @override
    def __enter__(self) -> Self:
        _ = super().__enter__()
        if self.sending.acquire(blocking=False):
            return self
        raise RuntimeError("LLMChat already has an active exchange")

    @override
    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        with self.lock:
            self.active.clear()
        self.sending.release()
        super().__exit__(exc_type, exc, traceback)

    @override
    def guard(self) -> GateLock:
        return self.lock

    @override
    def busy(self) -> bool:
        return bool(self.owner) or self.sending.locked() or bool(self.active)

    @override
    def reserve(self, owner: object) -> None:
        self.owner.append(owner)

    @override
    def release(self, owner: object) -> None:
        if self.owner and self.owner[0] is owner:
            self.owner.clear()

    @override
    def cancel(self) -> None:
        with self.lock:
            if self.active:
                self.active[0].set()

    @override
    def signal(self, event: Event) -> None:
        self.active.clear()
        self.active.append(event)

    @override
    def complete(self, event: Event, response: Message) -> None:
        if event.is_set():
            raise Cancelled(response)
        self.active.clear()
