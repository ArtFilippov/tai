"""Conversation boundary for the two independent LLM roles."""

from typing import Protocol

from garak.attempt import Conversation


class Chat(Protocol):
    """A conversational participant that answers the supplied history."""

    def description(self) -> dict[str, object]:
        ...

    def reply(self, conversation: Conversation) -> str:
        ...

    def prefill(self, conversation: Conversation, prefix: str) -> str:
        """Continue a partial assistant response.

        Args:
            conversation (Conversation): Complete request history.
            prefix (str): Assistant text already supplied.

        Returns:
            str: Complete assistant message including prefix.
        """
        ...
