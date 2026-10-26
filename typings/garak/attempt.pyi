"""Text-only portion of the pinned garak attempt objects."""

class Message:
    text: str
    lang: str | None
    notes: dict[str, object]
    def __init__(self, text: str, lang: str | None = ..., notes: dict[str, object] = ...) -> None: ...

class Turn:
    role: str
    content: Message
    def __init__(self, role: str, content: Message) -> None: ...

class Conversation:
    turns: list[Turn]
    def __init__(self, turns: list[Turn]) -> None: ...
    def last_message(self, role: str | None = ...) -> Message: ...

class Attempt:
    prompt: Conversation
    conversations: list[Conversation]
    notes: dict[str, object]
    outputs: list[Message]
