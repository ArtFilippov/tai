"""The conversation and participant-reservation contract implemented by chat adapters."""

from abc import abstractmethod
from threading import Event
from typing import Protocol

from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.messages import Message

from .chat_turn_contract import ChatTurn
from .exchange_gate_contract import ExchangeGate


class LLMChat(Protocol):
    """Own one model conversation, including its opaque checkpoints.

    Attributes:
        identifier (str): Model identity shared by detached diagnostics.
        policy (DiagnosticRedactor): Policy protecting diagnostic values.
    """

    identifier: str
    policy: DiagnosticRedactor

    def __init__(self, model: str, redactor: DiagnosticRedactor) -> None:
        """Retain the model identifier and diagnostic policy for this chat.

        Args:
            model (str): Model identifier for reproducibility.
            redactor (DiagnosticRedactor): Policy for detached diagnostics.
        """
        super().__init__()
        self.identifier, self.policy = model, redactor

    @property
    def model(self) -> str:
        """Return the model identifier for this conversation.

        Returns:
            str: Model identifier.
        """
        return self.identifier

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
    def redactor(self) -> DiagnosticRedactor:
        """Return the policy for detached diagnostic values.

        Returns:
            DiagnosticRedactor: Redaction policy of this conversation.
        """
        return self.policy

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
