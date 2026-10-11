"""Synchronous model exchanges with history owned by each LLMChat."""

from collections.abc import Callable
from threading import Event
from typing import override

from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.messages import Message

from ..chat_contract import LLMChat as LLMChatProtocol
from ..chat_turn_contract import ChatTurn
from ..conversation_history import ConversationHistory
from ..exchange_gate_contract import ExchangeGate
from ..generation_contract import Generation
from .conversation_state import ConversationState
from .cancelled import Cancelled
from .gate import Gate
from .turn import Turn


class LLMChat(LLMChatProtocol):
    """A text model callable receives an immutable, complete request history.

    Use independent instances for each role, even with one model.
    The callable performs one generation and returns its complete text.

    Args:
        generate (Callable[[tuple[ChatTurn, ...]], str]): Complete text generation.
        initial (Message | None, default=None): Prepared system turn, when present.
        model (str, default='custom'): Model identifier for reproducibility.
        system (str, default=''): Instructions retained across resets.
        redactor (DiagnosticRedactor): Policy for detached diagnostics.

    Raises:
        ValueError: System instructions were not paired with a prepared Message.
    """

    def __init__(
        self,
        generate: Callable[[tuple[ChatTurn, ...]], str],
        *,
        initial: Message | None = None,
        model: str = "custom",
        system: str = "",
        redactor: DiagnosticRedactor,
    ) -> None:
        if initial is None:
            if system:
                raise ValueError("System text requires a prepared Message")
        super().__init__(model, redactor)
        self.generation: Generation = generate
        self.dialogue: ConversationHistory = ConversationState(initial)
        self.gate: ExchangeGate = Gate()

    def __call__(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        return self.generation(turns)

    @override
    def cancel(self) -> None:
        """Request cancellation without waiting or changing conversation history."""
        self.gate.cancel()

    @override
    def history(self) -> tuple[ChatTurn, ...]:
        with self.gate.guard():
            return self.dialogue.history()

    @property
    @override
    def conversation_id(self) -> str:
        """Identity of the current conversation; only a successful reset rotates it."""
        with self.gate.guard():
            return self.dialogue.identity()

    @override
    def settings(self) -> dict[str, object] | list[object]:
        """Return a detached, JSON-serializable snapshot for run reproducibility.

        Transport subclasses include their configuration without credential
        values. The run records this snapshot before its first exchange.

        Returns:
            dict[str, object] | list[object]: Current model and completed conversation.
        """
        return self.redactor.structured(self.settings_data())

    def settings_data(self) -> dict[str, object]:
        with self.gate.guard():
            conversation_id, turns = self.dialogue.identity(), self.dialogue.history()
        return {
            "model": self.model,
            "conversation_id": conversation_id,
            "initial_history": [
                {"role": turn.role, "message": turn.message.snapshot()}
                for turn in turns
            ],
        }

    @override
    def diagnostic_settings(
        self, redactor: DiagnosticRedactor,
    ) -> dict[str, object] | list[object]:
        return redactor.structured(self.settings())

    @override
    def checkpoint(self) -> object:
        """Snapshot completed exchanges in this conversation as an opaque token.

        During send, the snapshot contains only the previously completed history.
        Restoring another branch does not invalidate this snapshot; reset does.

        Returns:
            object: Identity token owned by this conversation.
        """
        with self.gate.guard():
            return self.dialogue.checkpoint()

    @override
    def restore(self, checkpoint: object) -> None:
        """Restore a snapshot from this chat's current conversation.

        Foreign/stale tokens raise ValueError; active send or cancellation cleanup
        raises RuntimeError. Finish or cancel and await that send before retrying;
        no conversation mutation occurs on failure.

        Args:
            checkpoint (object): Opaque token previously returned by checkpoint().
        """
        with self.gate, self.gate.guard():
            self.dialogue.restore(checkpoint)

    @override
    def reset(self) -> None:
        """Start a clean conversation, preserving system instructions.

        All earlier checkpoints become stale. An active send must finish (or be
        cancelled and awaited) first; reset never clears an active stop signal.
        """
        with self.gate, self.gate.guard():
            self.dialogue.reset()

    @override
    def send(self, message: Message, *, cancellation: Event | None = None) -> Message:
        """Complete one exchange, optionally with explicit per-call cancellation.

        User.stop() closes admission between calls and does not set this signal.

        Args:
            message (Message): Next user turn.
            cancellation (Event | None, default=None): Per-call cancellation signal.

        Returns:
            Message: Complete response committed to the conversation.

        Raises:
            Cancelled: Cancellation was observed before admission or commit.
        """
        with self.gate:
            signal = cancellation if cancellation is not None else Event()
            with self.gate.guard():
                if signal.is_set():
                    raise Cancelled()
                self.gate.signal(signal)
                request = self.dialogue.history() + (Turn(("user", message)),)
            if signal.is_set():
                raise Cancelled()
            response = message.spawn(self(request, signal))
            with self.gate.guard():
                self.gate.complete(signal, response)
                self.dialogue.accept(request, response)
            return response

    @property
    @override
    def reservation(self) -> ExchangeGate:
        return self.gate
