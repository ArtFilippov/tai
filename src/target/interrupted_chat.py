"""A participant that fails after a supplied prefix of completed replies."""

from typing import override

from garak.attempt import Conversation

from target.chat import Chat


class InterruptedChat(Chat):
    """Keep completed exchanges real while exposing an injected boundary failure."""

    def __init__(self, replies: list[str], failure: str) -> None:
        """Bind the completed response prefix and subsequent failure.

        Args:
            replies (list[str]): Replies available before interruption.
            failure (str): Explanation of the interruption.
        """
        super().__init__()
        self.replies = replies
        self.failure = failure

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "interrupted_chat",
            "model": "controlled-target"
        }

    @override
    def reply(self, conversation: Conversation) -> str:
        if not self.replies:
            raise ValueError(self.failure)
        return self.replies.pop(0)

    @override
    def prefill(self, conversation: Conversation, prefix: str) -> str:
        return prefix + self.reply(conversation)
