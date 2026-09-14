"""Immutable text shared by conversations, strategies, and evaluators."""

import json
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Message:
    text: str
    id: str = field(default_factory=lambda: str(uuid4()), init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Message text must be a string")

    def base(self) -> str:
        return self.text

    def to_string(self) -> str:
        return self.text

    def dump(self) -> str:
        return json.dumps(
            {"id": self.id, "base": self.base(), "text": self.text},
            ensure_ascii=False,
        )
