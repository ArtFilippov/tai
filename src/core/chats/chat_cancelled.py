"""Cancellation preserves an uncommitted response as diagnostic evidence."""

from abc import ABC, abstractmethod

from core.primitives.messages import Message


class ChatCancelled(RuntimeError, ABC):
    """An exchange stopped without committing its response to the dialogue."""

    def __init__(
        self, response: Message | None = None, *, diagnostic: dict[str, object] | None = None,
    ) -> None:
        """Keep an optional observed response and detached transport diagnostic.

        Args:
            response (Message | None, default=None): Uncommitted response.
            diagnostic (dict[str, object] | None, default=None): Detached observations.
        """
        super().__init__("LLMChat exchange cancelled")
        self.observation: tuple[Message | None, dict[str, object] | None] = (
            response,
            dict(diagnostic) if diagnostic is not None else None
        )

    @property
    def response(self) -> Message | None:
        """Return the observed, uncommitted Message when available.

        Returns:
            Message | None: Uncommitted response, if any.
        """
        return self.observation[0]

    @property
    def diagnostic(self) -> dict[str, object] | None:
        """Return detached transport observations when available.

        Returns:
            dict[str, object] | None: Incomplete transport details, if any.
        """
        return self.observation[1]

    @abstractmethod
    def observed(self) -> Message | None:
        """Return a complete but uncommitted response when one was received.

        Returns:
            Message | None: The observed response, if any.
        """
