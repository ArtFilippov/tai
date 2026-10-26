"""Contract for assessing independent Target candidates."""

from collections.abc import Iterable, Mapping
from typing import Protocol

from garak.attempt import Message

from attack.assessment import Assessment
from target.chat import Chat


class CandidateRun(Protocol):
    """Present each completed reply and stop at the first positive assessment."""

    def execute(self, candidates: Iterable[Message], target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        ...
