"""Public boundary for observable tense-reformulation preparation."""

from collections.abc import Mapping
from typing import Protocol

from attack.probe import Probe


class Phrasing(Probe, Protocol):
    """Expose a request-bound attack with journaled research reformulation."""

    def reformulate(self, candidate: Mapping[str, str | int]) -> str:
        ...
