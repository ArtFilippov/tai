"""Shared weights and cancellable serialization of local generations."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import override
from weakref import WeakValueDictionary

from .conversation import ChatTurn
from .qwen_config import Generation, LocalCheckpoint
from .qwen_contracts import LocalRuntimeProtocol, QwenExecutorProtocol
from .qwen_runtime import check_cancelled, load_runtime


@dataclass
class Executor(QwenExecutorProtocol):
    """Weights and an execution gate only; no history, KV-cache, or stop signal.

    Attributes:
        device (str): Explicit CUDA device.
        dtype (str): Requested weight dtype.
        gate (Lock, default=field(default_factory=Lock)): Generation serialization lock.
        _loaded (LocalRuntimeProtocol | None, default=None): Atomically published complete runtime.
    """

    device: str
    dtype: str
    gate: Lock = field(default_factory=Lock)
    _loaded: LocalRuntimeProtocol | None = None

    @contextmanager
    def _admitted(self, cancellation: Event) -> Generator[None]:
        check_cancelled(cancellation)
        for _ in iter(lambda: self.gate.acquire(timeout=0.02), True):
            check_cancelled(cancellation)
        try:
            yield
        finally:
            self.gate.release()

    @override
    def actual_dtype(self) -> str | None:
        return self._loaded.actual_dtype() if self._loaded is not None else None

    @override
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @override
    def generate(
        self, turns: tuple[ChatTurn, ...], cancellation: Event,
        generation: Generation, *, checkpoint: LocalCheckpoint,
        importer: Callable[[str], object],
    ) -> str:
        with self._admitted(cancellation):
            check_cancelled(cancellation)
            if self._loaded is None:
                self._loaded = load_runtime(checkpoint, self.dtype, cancellation, importer)
            check_cancelled(cancellation)
            return self._loaded.generate(turns, cancellation, generation, device=self.device)


_executors: WeakValueDictionary[tuple[str, str, str, str], Executor] = WeakValueDictionary()
_executors_lock = Lock()


def shared_executor(checkpoint: LocalCheckpoint, dtype: str) -> Executor:
    key = checkpoint.key(dtype)
    with _executors_lock:
        if (executor := _executors.get(key)) is None:
            executor = Executor(checkpoint.device, dtype)
            _executors[key] = executor
        return executor
