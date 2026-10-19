"""Deterministic conversation participant for weight-free runs."""

from typing import override

from garak.attempt import Conversation

from fitd.chat import Chat


class ScriptedChat(Chat):
    """Replies in order while retaining the histories it was given."""

    def __init__(self, replies: list[str]) -> None:
        """Keep the provided replies in order.

        Args:
            replies (list[str]): Responses for successive requests.
        """
        super().__init__()
        self.replies: list[str] = replies
        self.histories: list[Conversation] = []

    @override
    def reply(self, conversation: Conversation) -> str:
        self.histories.append(conversation)
        return self.replies.pop(0)
