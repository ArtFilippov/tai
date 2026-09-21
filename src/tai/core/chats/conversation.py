"""Synchronous model exchanges with history owned by each LLMChat."""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from threading import Event, Lock
from typing import Literal, Self
from uuid import uuid4

from tai.core.primitives.diagnostics import DiagnosticObject, DiagnosticRedactor
from tai.core.primitives.messages import Message

type ModelRole = Literal["Target", "Attacker", "Evaluator"]


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: Literal["system", "user", "assistant"]
    message: Message


class ChatCancelled(RuntimeError):
    """An exchange was cancelled; an observed response is diagnostic only."""

    def __init__(
        self,
        response: Message | None = None,
        *,
        diagnostic: dict[str, object] | None = None,
    ) -> None:
        super().__init__("LLMChat exchange cancelled")
        self.response = response
        # Incomplete transport data is separate from an observed full Message.
        self.diagnostic = diagnostic.copy() if diagnostic is not None else None


class LLMChat:
    """A text model callable receives an immutable, complete request history.

    Use independent instances for each role, even with one model.
    The callable performs one generation and returns its complete text.
    """

    def __init__(
        self,
        generate: Callable[[tuple[ChatTurn, ...]], str],
        *,
        model: str = "custom",
        system: str = "",
    ) -> None:
        self.model = model
        self._generate_callable: Callable[[tuple[ChatTurn, ...], Event], str] = (
            lambda turns, cancellation: generate(turns)
        )
        self._turns: tuple[ChatTurn, ...] = (
            (ChatTurn("system", Message(system)),) if system else ()
        )
        self._system_turns = self._turns
        self._conversation_id = str(uuid4())
        self._checkpoints: dict[object, tuple[ChatTurn, ...]] = {}
        self._lock = Lock()
        self._sending = Lock()
        self._active: Event | None = None
        self._owner: object | None = None
        self._redactor = DiagnosticRedactor()

    @classmethod
    def cancellable(
        cls,
        generate: Callable[[tuple[ChatTurn, ...], Event], str],
        *,
        model: str = "custom",
        system: str = "",
    ) -> Self:
        """Use a generator that observes a per-exchange cancellation Event.

        The generator must finish its I/O/cleanup before returning or raising
        ChatCancelled. It must not clear the signal or retain it for later calls.
        """
        chat = cls(lambda turns: "", model=model, system=system)
        chat._generate_callable = generate
        return chat

    def _generate(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        # Suppress a callback when cancellation arrived after admission. Once
        # invoked, an uncooperative callable must return before send can unwind.
        if cancellation.is_set():
            raise ChatCancelled()
        return self._generate_callable(turns, cancellation)

    def cancel(self) -> None:
        """Request cancellation without waiting or changing conversation history."""
        with self._lock:
            if self._active is not None:
                self._active.set()

    def history(self) -> tuple[ChatTurn, ...]:
        with self._lock:
            return self._turns

    @property
    def conversation_id(self) -> str:
        """Identity of the current conversation; only a successful reset rotates it."""
        with self._lock:
            return self._conversation_id

    def settings(self) -> DiagnosticObject:
        """Return a detached, JSON-serializable snapshot for run reproducibility.

        Transport subclasses include their configuration without credential
        values. The run records this snapshot before its first exchange.
        """
        return self._settings()

    def _settings(self) -> dict[str, object]:
        import json

        with self._lock:
            conversation_id, turns = self._conversation_id, self._turns
        return {
            "model": self.model,
            "conversation_id": conversation_id,
            "initial_history": [
                {"role": turn.role, "message": json.loads(turn.message.dump())}
                for turn in turns
            ],
        }

    def _diagnostic_settings(self, redactor: DiagnosticRedactor) -> DiagnosticObject:
        return redactor.snapshot(self.settings())

    def checkpoint(self) -> object:
        """Snapshot completed exchanges in this conversation as an opaque token.

        During send, the snapshot contains only the previously completed history.
        Restoring another branch does not invalidate this snapshot; reset does.
        """
        with self._lock:
            token = object()
            self._checkpoints[token] = self._turns
            return token

    def restore(self, checkpoint: object) -> None:
        """Restore a snapshot from this chat's current conversation.

        Raises ValueError for foreign/stale tokens and RuntimeError while send is
        still active, including cancellation cleanup. Finish or cancel and await
        that send before retrying; no conversation mutation occurs on failure.
        """
        if not self._sending.acquire(blocking=False):
            raise RuntimeError("LLMChat already has an active exchange")
        try:
            with self._lock:
                if (
                    type(checkpoint) is not object
                    or checkpoint not in self._checkpoints
                ):
                    raise ValueError("Foreign or stale LLMChat checkpoint")
                self._turns = self._checkpoints[checkpoint]
        finally:
            self._sending.release()

    def reset(self) -> None:
        """Start a clean conversation, preserving system instructions.

        All earlier checkpoints become stale. An active send must finish (or be
        cancelled and awaited) first; reset never clears an active stop signal.
        """
        if not self._sending.acquire(blocking=False):
            raise RuntimeError("LLMChat already has an active exchange")
        try:
            with self._lock:
                self._conversation_id = str(uuid4())
                self._turns = self._system_turns
                self._checkpoints.clear()
        finally:
            self._sending.release()

    def send(self, message: Message, *, cancellation: Event | None = None) -> Message:
        """Complete one exchange, optionally with explicit per-call cancellation.

        User.stop() closes admission between calls and does not set this signal.
        """
        if not self._sending.acquire(blocking=False):
            raise RuntimeError("LLMChat already has an active exchange")
        try:
            cancellation = cancellation if cancellation is not None else Event()
            with self._lock:
                if cancellation.is_set():
                    raise ChatCancelled()
                self._active = cancellation
                request = self._turns + (ChatTurn("user", message),)
            response = message.spawn(self._generate(request, cancellation))
            with self._lock:
                if cancellation.is_set():
                    raise ChatCancelled(response)
                self._turns = request + (ChatTurn("assistant", response),)
                self._active = None
            return response
        finally:
            with self._lock:
                self._active = None
            self._sending.release()


def _acquire_chats(chats: tuple[LLMChat, ...], owner: object) -> None:
    """Reserve a whole participant set before changing either conversation."""
    with ExitStack() as locks:
        for chat in sorted(chats, key=id):
            locks.enter_context(chat._lock)
        if any(chat._owner is not None or chat._sending.locked() for chat in chats):
            raise RuntimeError("LLMChat already belongs to an active execution")
        for chat in chats:
            chat._owner = owner


def _release_chats(chats: tuple[LLMChat, ...], owner: object) -> None:
    """Release only this generation's reservations, together and after cleanup."""
    with ExitStack() as locks:
        for chat in sorted(chats, key=id):
            locks.enter_context(chat._lock)
        for chat in chats:
            if chat._owner is owner:
                chat._owner = None
