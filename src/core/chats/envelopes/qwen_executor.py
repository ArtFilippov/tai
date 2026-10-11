"""Shared weights and cancellable serialization of local generations."""

from collections.abc import Callable, Mapping
from threading import Event, Lock
from typing import override

from ..chat_turn_contract import ChatTurn
from ..qwen_checkpoint_contract import LocalCheckpointPort
from ..qwen_runtime_contract import LocalRuntimeProtocol
from ..qwen_executor_contract import QwenExecutorProtocol
from .qwen_gate_utils import admitted, check_cancelled
from .qwen_runtime_utils import load_runtime


class Executor(QwenExecutorProtocol):
    """Weights and an execution gate only; no history, KV-cache, or stop signal."""

    def __init__(self, device: str, dtype: str) -> None:
        """Prepare the shared gate for one local model profile.

        Args:
            device (str): Explicit CUDA device.
            dtype (str): Requested weight dtype.
        """
        super().__init__()
        self.profile = (device, dtype)
        self.gate: Lock = Lock()
        self.loaded: list[LocalRuntimeProtocol] = []

    @property
    @override
    def device(self) -> str:
        return self.profile[0]

    @property
    @override
    def dtype(self) -> str:
        return self.profile[1]

    @override
    def actual_dtype(self) -> str | None:
        return self.loaded[0].actual_dtype() if self.loaded else None

    @override
    def is_loaded(self) -> bool:
        return bool(self.loaded)

    def runtime(
        self, checkpoint: LocalCheckpointPort, cancellation: Event,
        importer: Callable[[str], object],
    ) -> LocalRuntimeProtocol:
        if not self.loaded:
            self.loaded.append(load_runtime(checkpoint, self.profile[1], cancellation, importer))
        return self.loaded[0]

    @override
    def score_continuation(
        self, prompt: str, expected: str, system: str, *, checkpoint: LocalCheckpointPort,
        importer: Callable[[str], object],
    ) -> float:
        with admitted(self.gate, Event()):
            return self.runtime(checkpoint, Event(), importer).score_continuation(
                prompt, expected, system, device=self.profile[0],
            )

    @override
    def generate(
        self, turns: tuple[ChatTurn, ...], cancellation: Event,
        generation: Mapping[str, int | float | bool], *, checkpoint: LocalCheckpointPort,
        importer: Callable[[str], object],
    ) -> str:
        with admitted(self.gate, cancellation):
            check_cancelled(cancellation)
            runtime = self.runtime(checkpoint, cancellation, importer)
            check_cancelled(cancellation)
            return runtime.generate(
                turns, cancellation, generation, device=self.profile[0],
            )
