"""Boundary for a configured FITD strategy."""

from collections.abc import Mapping
from typing import Protocol

from attack.assessment import Assessment
from target.chat import Chat


class Probe(Protocol):
    """Produces a presented Target response from one FITD conversation."""

    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        ...
