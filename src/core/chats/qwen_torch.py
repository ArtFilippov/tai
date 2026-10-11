"""Torch operations used by the history-free local runtime."""

from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import Protocol

from .qwen_cuda import Cuda


class Torch(Protocol):
    """Dtypes, inference scope, and the CUDA device selected by the chat."""

    @property
    def cuda(self) -> Cuda: ...

    @property
    def float16(self) -> object: ...

    @property
    def bfloat16(self) -> object: ...

    @property
    def float32(self) -> object: ...

    @abstractmethod
    def inference_mode(self) -> AbstractContextManager[object]: ...
