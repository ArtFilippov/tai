"""One Attacker history across independent FITD strategy checks."""

from dataclasses import dataclass
from typing import override

from garak.attempt import Conversation, Message, Turn

from target.chat import Chat


@dataclass(eq=False)
class ContinuingChat(Chat):
    """Accumulates the Attacker's prompts and replies separately from Target.

    Attributes:
        chat (Chat): Participant receiving the accumulated conversation.
        turns (list[Turn]): Completed conversation turns.
    """

    chat: Chat
    turns: list[Turn]

    def __init__(self, chat: Chat) -> None:
        """Begin an independent Attacker conversation.

        Args:
            chat (Chat): Participant that generates responses.
        """
        super().__init__()
        self.chat = chat
        self.turns = []

    @override
    def reply(self, conversation: Conversation) -> str:
        self.turns.extend(conversation.turns)
        response = self.chat.reply(
            Conversation(
                list(self.turns)
            )
        )
        self.turns.append(
            Turn(
                "assistant",
                Message(text=response)
            )
        )
        return response
