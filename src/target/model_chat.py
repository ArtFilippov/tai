"""Independent conversations backed by any shared model."""

from dataclasses import dataclass
from typing import override

from garak.attempt import Conversation

from target.chat import Chat
from target.model import Model


@dataclass(eq=False)
class ModelChat(Chat):
    """Converts only the supplied conversation into a model request.

    Attributes:
        model (Model): Shared history-free generator.
    """

    model: Model

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "model_chat",
            "model": self.model.settings()
        }

    @override
    def reply(self, conversation: Conversation) -> str:
        return self.model.generate(
            [
                {
                    "role": turn.role,
                    "content": turn.content.text
                }
                for turn in conversation.turns
            ]
        )

    @override
    def prefill(self, conversation: Conversation, prefix: str) -> str:
        return self.model.prefill(
            [
                {
                    "role": turn.role,
                    "content": turn.content.text
                }
                for turn in conversation.turns
            ],
            prefix
        )
