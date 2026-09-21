"""Native text/conversation values; absent text and prompts remain nullable."""

from uuid import UUID

class Message:
    text: str | None
    lang: str | None
    def __init__(self, text: str | None = ..., lang: str | None = ...) -> None: ...

class Turn:
    role: str
    content: Message
    def __init__(self, role: str, content: Message) -> None: ...

class Conversation:
    turns: list[Turn]
    def __init__(self, turns: list[Turn] = ..., notes: dict[str, object] | None = ...) -> None: ...
    def last_message(self, role: str | None = ...) -> Message: ...

class Attempt:
    uuid: UUID
    notes: dict[str, object]
    conversations: list[Conversation]
    @property
    def prompt(self) -> Conversation | None: ...
