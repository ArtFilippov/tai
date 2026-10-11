"""One independently cancellable scripted model conversation."""

from threading import Event
from typing import override

from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.messages import Message

from ..chat_turn_contract import ChatTurn
from ..event_generation import EventGeneration
from .conversation import LLMChat


class CancellableChat(LLMChat):
    """Give a model callback the cancellation event for this exchange.

    Args:
        generate (EventGeneration): Event-aware response callback.
        initial (Message | None, default=None): Prepared system Message.
        model (str, default='custom'): Model identifier.
        system (str, default=''): Instructions retained across resets.
        redactor (DiagnosticRedactor): Policy for detached diagnostics.
    """

    def __init__(
        self, generate: EventGeneration, *, initial: Message | None = None,
        model: str = "custom", system: str = "", redactor: DiagnosticRedactor,
    ) -> None:
        super().__init__(
            lambda turns: "", initial=initial, model=model, system=system, redactor=redactor,
        )
        self.callback: EventGeneration = generate

    def invoke(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        """Finish generation with the current exchange's stop signal.

        Args:
            turns (tuple[ChatTurn, ...]): Complete immutable request history.
            cancellation (Event): Signal belonging only to this exchange.

        Returns:
            str: Complete response text.
        """
        return self.callback(turns, cancellation)

    @override
    def __call__(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        return self.invoke(turns, cancellation)
