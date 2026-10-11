"""The conversation and participant-reservation contract implemented by chat adapters."""

from abc import abstractmethod
from collections.abc import Callable
from threading import Event
from typing import Protocol, Self

from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.messages import Message

from .chat_state import ChatTurn, ExchangeGate


class LLMChatProtocol(Protocol):
    model: str

    @classmethod
    @abstractmethod
    def cancellable(
        cls, generate: Callable[[tuple[ChatTurn, ...], Event], str], *,
        create: Callable[[str], Message],
        redactor: DiagnosticRedactor,
        model: str = "custom", system: str = "",
    ) -> Self: ...

    @abstractmethod
    def cancel(self) -> None: ...

    @abstractmethod
    def history(self) -> tuple[ChatTurn, ...]: ...

    @property
    @abstractmethod
    def conversation_id(self) -> str: ...

    @abstractmethod
    def settings(self) -> dict[str, object] | list[object]: ...

    @property
    @abstractmethod
    def redactor(self) -> DiagnosticRedactor: ...

    @abstractmethod
    def diagnostic_settings(
        self, redactor: DiagnosticRedactor,
    ) -> dict[str, object] | list[object]: ...

    @abstractmethod
    def checkpoint(self) -> object: ...

    @abstractmethod
    def restore(self, checkpoint: object) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def send(self, message: Message, *, cancellation: Event | None = None) -> Message: ...

    @property
    @abstractmethod
    def reservation(self) -> ExchangeGate: ...
