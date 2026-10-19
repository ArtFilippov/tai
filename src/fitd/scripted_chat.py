"""Deterministic conversation participant for weight-free runs."""

from dataclasses import dataclass
from typing import override

from garak.attempt import Conversation

from fitd.chat import Chat


@dataclass(eq=False)
class ScriptedChat(Chat):
    """Replies in order while retaining the histories it was given.

    Attributes:
        replies (list[str]): Responses for successive requests.
        histories (list[Conversation]): Conversations supplied to this role.
    """

    replies: list[str]
    histories: list[Conversation]

    def __init__(self, replies: list[str]) -> None:
        """Keep the provided replies in order.

        Args:
            replies (list[str]): Responses for successive requests.
        """
        super().__init__()
        self.replies = replies
        self.histories = []

    @override
    def reply(self, conversation: Conversation) -> str:
        self.histories.append(conversation)
        return self.replies.pop(0)
