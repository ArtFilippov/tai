"""Synchronous model exchanges with history owned by each LLMChat."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Literal

from tai.messages import Message

type ModelRole = Literal["Target", "Attacker"]


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: Literal["system", "user", "assistant"]
    message: Message


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
        self._generate = generate
        self._turns: tuple[ChatTurn, ...] = (
            (ChatTurn("system", Message(system)),) if system else ()
        )
        self._lock = Lock()
        self._sending = Lock()

    def history(self) -> tuple[ChatTurn, ...]:
        with self._lock:
            return self._turns

    def send(self, message: Message) -> Message:
        if not self._sending.acquire(blocking=False):
            raise RuntimeError("LLMChat already has an active exchange")
        try:
            request = self.history() + (ChatTurn("user", message),)
            response = Message(self._generate(request))
            with self._lock:
                self._turns = request + (ChatTurn("assistant", response),)
            return response
        finally:
            self._sending.release()
