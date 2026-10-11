"""Share weights by pinned model profile without sharing conversation state."""

from threading import Lock
from weakref import WeakValueDictionary

from ..qwen_checkpoint_contract import LocalCheckpointPort
from .qwen_executor import Executor


executors: WeakValueDictionary[tuple[str, str, str, str], Executor] = WeakValueDictionary()
executors_lock = Lock()


def shared_executor(checkpoint: LocalCheckpointPort, dtype: str, device: str) -> Executor:
    key = checkpoint.key(dtype)
    with executors_lock:
        if (executor := executors.get(key)) is None:
            executor = Executor(device, dtype)
            executors[key] = executor
        return executor
