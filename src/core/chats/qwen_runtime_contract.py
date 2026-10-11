"""History-free local inference on shared weights."""

from collections.abc import Mapping
from threading import Event
from typing import Protocol

from .chat_turn_contract import ChatTurn


class LocalRuntimeProtocol(Protocol):
    """Generate complete text or score an explicit continuation."""

    def score_continuation(
        self, prompt: str, expected: str, system: str, *, device: str,
    ) -> float: ...

    def actual_dtype(self) -> str: ...

    def generate(
        self, turns: tuple[ChatTurn, ...], cancellation: Event,
        generation: Mapping[str, int | float | bool], *, device: str,
    ) -> str: ...
