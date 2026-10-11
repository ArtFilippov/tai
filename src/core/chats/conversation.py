"""Synchronous model exchanges with history owned by each LLMChat."""

from __future__ import annotations

import json
from collections.abc import Callable
from threading import Event
from typing import Literal, Self, override

from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.messages import Message

from .chat_state import ChatTurn, ConversationState, ExchangeGate
from .chat_contract import LLMChatProtocol

__all__ = ["ChatTurn", "ChatCancelled", "LLMChat", "ModelRole", "acquire_chats", "release_chats"]

type ModelRole = Literal["Target", "Attacker", "Evaluator"]


class ChatCancelled(RuntimeError):
    """An exchange was cancelled; an observed response is diagnostic only.

    Args:
        response (Message | None, default=None): Fully observed, uncommitted response.
        diagnostic (dict[str, object] | None, default=None): Incomplete transport observations.
    """

    def __init__(
        self,
        response: Message | None = None,
        *,
        diagnostic: dict[str, object] | None = None,
    ) -> None:
        super().__init__("LLMChat exchange cancelled")
        self.response = response
        self.diagnostic = diagnostic.copy() if diagnostic is not None else None


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
        super().__init__()
        self.model = model
        self._generate_callable: Callable[[tuple[ChatTurn, ...], Event], str] = (
            lambda turns, cancellation: generate(turns)
        )
        if system and initial is None:
            raise ValueError("System text requires a prepared Message")
        turns = (ChatTurn("system", initial),) if initial is not None else ()
        self._history = ConversationState(turns, turns)
        self._execution = ExchangeGate()
        self._redactor: DiagnosticRedactor = redactor

    @classmethod
    @override
    def cancellable(
        cls,
        generate: Callable[[tuple[ChatTurn, ...], Event], str],
        *,
        create: Callable[[str], Message],
        model: str = "custom",
        system: str = "",
        redactor: DiagnosticRedactor,
    ) -> Self:
        """Use a generator that observes a per-exchange cancellation Event.

        The generator must finish its I/O/cleanup before returning or raising
        ChatCancelled. It must not clear the signal or retain it for later calls.

        Args:
            generate (Callable[[tuple[ChatTurn, ...], Event], str]): Cancellable model call.
            create (Callable[[str], Message]): Assembled textual result constructor.
            model (str, default='custom'): Model identifier.
            system (str, default=''): Instructions retained across resets.
            redactor (DiagnosticRedactor): Policy for detached diagnostics.

        Returns:
            Self: Independent conversation using the supplied callable.
        """
        chat = cls(
            lambda turns: "", initial=create(system) if system else None,
            model=model, system=system, redactor=redactor,
        )
        chat._generate_callable = generate
        return chat

    def _generate(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        if cancellation.is_set():
            raise ChatCancelled()
        return self._generate_callable(turns, cancellation)

    @override
    def cancel(self) -> None:
        """Request cancellation without waiting or changing conversation history."""
        self._execution.cancel()

    @override
    def history(self) -> tuple[ChatTurn, ...]:
        with self._execution.lock:
            return self._history.turns

    @property
    @override
    def conversation_id(self) -> str:
        """Identity of the current conversation; only a successful reset rotates it."""
        with self._execution.lock:
            return self._history.identity

    @override
    def settings(self) -> dict[str, object] | list[object]:
        """Return a detached, JSON-serializable snapshot for run reproducibility.

        Transport subclasses include their configuration without credential
        values. The run records this snapshot before its first exchange.

        Returns:
            dict[str, object] | list[object]: Current model and completed conversation.
        """
        return self._redactor.structured(self._settings())

    def _settings(self) -> dict[str, object]:
        with self._execution.lock:
            conversation_id, turns = self._history.identity, self._history.turns
        return {
            "model": self.model,
            "conversation_id": conversation_id,
            "initial_history": [
                {"role": turn.role, "message": json.loads(turn.dump())}
                for turn in turns
            ],
        }

    def _diagnostic_settings(
        self, redactor: DiagnosticRedactor,
    ) -> dict[str, object] | list[object]:
        return redactor.structured(self.settings())

    @property
    @override
    def redactor(self) -> DiagnosticRedactor:
        return self._redactor

    @override
    def diagnostic_settings(
        self, redactor: DiagnosticRedactor,
    ) -> dict[str, object] | list[object]:
        return self._diagnostic_settings(redactor)

    @override
    def checkpoint(self) -> object:
        """Snapshot completed exchanges in this conversation as an opaque token.

        During send, the snapshot contains only the previously completed history.
        Restoring another branch does not invalidate this snapshot; reset does.

        Returns:
            object: Identity token owned by this conversation.
        """
        with self._execution.lock:
            token = object()
            self._history.checkpoints[token] = self._history.turns
            return token

    @override
    def restore(self, checkpoint: object) -> None:
        """Restore a snapshot from this chat's current conversation.

        Foreign/stale tokens raise ValueError; active send or cancellation cleanup
        raises RuntimeError. Finish or cancel and await that send before retrying;
        no conversation mutation occurs on failure.

        Args:
            checkpoint (object): Opaque token previously returned by checkpoint().
        """
        with self._execution, self._execution.lock:
            self._history.restore(checkpoint)

    @override
    def reset(self) -> None:
        """Start a clean conversation, preserving system instructions.

        All earlier checkpoints become stale. An active send must finish (or be
        cancelled and awaited) first; reset never clears an active stop signal.
        """
        with self._execution, self._execution.lock:
            self._history.reset()

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
            ChatCancelled: Cancellation was observed before admission or commit.
        """
        with self._execution:
            signal = cancellation if cancellation is not None else Event()
            with self._execution.lock:
                if signal.is_set():
                    raise ChatCancelled()
                self._execution.active = signal
                request = self._history.turns + (ChatTurn("user", message),)
            response = message.spawn(self._generate(request, signal))
            with self._execution.lock:
                if signal.is_set():
                    raise ChatCancelled(response)
                self._history.turns = request + (ChatTurn("assistant", response),)
                self._execution.active = None
            return response

    @property
    @override
    def reservation(self) -> ExchangeGate:
        return self._execution


def acquire_chats(chats: tuple[LLMChat, ...], owner: object) -> None:
    """Reserve a whole participant set before changing either conversation.

    Args:
        chats (tuple[LLMChat, ...]): Distinct conversation participants.
        owner (object): Identity of the execution reserving the participants.
    """
    gates = tuple(chat.reservation for chat in sorted(chats, key=id))
    ExchangeGate.acquire_all(gates, owner)


def release_chats(chats: tuple[LLMChat, ...], owner: object) -> None:
    """Release only this generation's reservations, together and after cleanup.

    Args:
        chats (tuple[LLMChat, ...]): Previously reserved participants.
        owner (object): Identity of the execution releasing the participants.
    """
    gates = tuple(chat.reservation for chat in sorted(chats, key=id))
    ExchangeGate.release_all(gates, owner)


# Compatibility for existing integrations while consumers migrate to the public API.
_acquire_chats = acquire_chats
_release_chats = release_chats
