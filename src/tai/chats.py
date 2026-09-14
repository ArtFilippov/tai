"""Synchronous model exchanges with history owned by each LLMChat."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock
from typing import Literal, Self

from tai.messages import Message

type ModelRole = Literal["Target", "Attacker"]


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: Literal["system", "user", "assistant"]
    message: Message


class ChatCancelled(RuntimeError):
    """An exchange was cancelled; an observed response is diagnostic only."""

    def __init__(self, response: Message | None = None) -> None:
        super().__init__("LLMChat exchange cancelled")
        self.response = response


class LLMChat:
    """A text model callable receives an immutable, complete request history.

    Use independent instances for Target and Attacker, even with one model.
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
        self._lock = Lock()
        self._sending = Lock()
        self._active: Event | None = None

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

    def send(self, message: Message, *, cancellation: Event | None = None) -> Message:
        """Complete one exchange, optionally governed by an owner's stop signal."""
        if not self._sending.acquire(blocking=False):
            raise RuntimeError("LLMChat already has an active exchange")
        try:
            cancellation = cancellation if cancellation is not None else Event()
            with self._lock:
                if cancellation.is_set():
                    raise ChatCancelled()
                self._active = cancellation
                request = self._turns + (ChatTurn("user", message),)
            response = Message(self._generate(request, cancellation))
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
