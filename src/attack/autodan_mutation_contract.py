"""Structural native mutation request boundary."""

from typing import Protocol

from garak.attempt import Conversation, Message


class AutoDANMutationContract(Protocol):
    """Provide a single separate Attacker reply for garak's mutation."""

    def generate(self, prompt: Conversation) -> list[Message]:
        ...

    def description(self) -> dict[str, object]:
        ...
