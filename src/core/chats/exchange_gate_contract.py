"""A chat's exchange, cancellation and participant reservation boundary."""

from threading import Event
from types import TracebackType
from typing import Protocol, Self

from core.primitives.messages import Message

from .gate_lock import GateLock


class ExchangeGate(Protocol):
    """Serialize a chat's exchanges and reserve it for a single execution."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        return None

    def guard(self) -> GateLock: ...

    def busy(self) -> bool: ...

    def reserve(self, owner: object) -> None: ...

    def release(self, owner: object) -> None: ...

    def cancel(self) -> None: ...

    def signal(self, event: Event) -> None: ...

    def complete(self, event: Event, response: Message) -> None: ...
