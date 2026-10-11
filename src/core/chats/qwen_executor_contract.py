"""A serialized shared model executor without conversation history."""

from collections.abc import Callable, Mapping
from threading import Event
from typing import Protocol

from .chat_turn_contract import ChatTurn
from .qwen_checkpoint_contract import LocalCheckpointPort


class QwenExecutorProtocol(Protocol):
    """Publish one loaded local runtime and preserve per-call cancellation."""

    @property
    def device(self) -> str: ...

    @property
    def dtype(self) -> str: ...

    def actual_dtype(self) -> str | None: ...

    def is_loaded(self) -> bool: ...

    def score_continuation(
        self, prompt: str, expected: str, system: str, *, checkpoint: LocalCheckpointPort,
        importer: Callable[[str], object],
    ) -> float: ...

    def generate(
        self, turns: tuple[ChatTurn, ...], cancellation: Event,
        generation: Mapping[str, int | float | bool], *, checkpoint: LocalCheckpointPort,
        importer: Callable[[str], object],
    ) -> str: ...
