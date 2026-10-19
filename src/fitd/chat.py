"""Conversation boundary for the two independent LLM roles."""

from typing import Protocol

from garak.attempt import Conversation


class Chat(Protocol):
    """A conversational participant that answers the supplied history."""

    def reply(self, conversation: Conversation) -> str:
        ...
