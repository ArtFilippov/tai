"""Expose pinned garak classes through the narrow declarations in native.pyi.

The adjacent stubs describe the used native surface and its real inheritance.
Runtime objects are the original garak classes, so virtual dispatch, native
adaptation and exception identity survive. Profile validation narrows native
optional fields at the boundary rather than declaring them non-null in a stub.
"""

from garak.attempt import Attempt, Conversation
from garak.attempt import Message as GarakMessage
from garak.detectors.mitigation import Prefixes
from garak.exception import GarakException
from garak.probes.fitd import FITD

__all__ = [
    "FITD", "Attempt", "Conversation", "GarakMessage", "Prefixes", "GarakException",
    "attempt_prompt", "message_text", "levels_completed",
]


def attempt_prompt(attempt: Attempt) -> Conversation:
    if (prompt := attempt.prompt) is None:
        raise ValueError("FITD requires a prepared native conversation")
    return prompt


def message_text(message: GarakMessage) -> str:
    if (text := message.text) is None:
        raise ValueError("The English FITD profile requires text messages")
    return text


def levels_completed(attempt: Attempt) -> int:
    value = attempt.notes["levels_completed"]
    if not isinstance(value, int):
        raise TypeError("FITD ladder progress must be an integer")
    return value
