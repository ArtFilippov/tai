"""Typed re-exports of the original pinned classes; no substitute implementations."""

from garak.attempt import Attempt as Attempt, Conversation as Conversation
from garak.attempt import Message as GarakMessage
from garak.detectors.mitigation import Prefixes as Prefixes
from garak.exception import GarakException as GarakException
from garak.probes.fitd import FITD as FITD

__all__ = [
    "FITD", "Attempt", "Conversation", "GarakMessage", "Prefixes", "GarakException",
    "attempt_prompt", "message_text", "levels_completed",
]

def attempt_prompt(attempt: Attempt) -> Conversation: ...
def message_text(message: GarakMessage) -> str: ...
def levels_completed(attempt: Attempt) -> int: ...
