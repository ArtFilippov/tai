"""Call-local tensors cleared before GPU synchronization."""

from typing import Protocol

from .qwen_input_batch import InputBatch
from .qwen_model import Model
from .qwen_tensor import Tensor
from .qwen_tokenizer import Tokenizer


class GenerationBuffersPort(Protocol):
    """Retain one request's token tensors only while generation is active."""

    def batch(self) -> InputBatch: ...

    def prepare(
        self, tokenizer: Tokenizer, model: Model,
        turns: list[dict[str, str]], maximum: int | float | bool,
    ) -> None: ...

    def move(self, device: str) -> None: ...

    def accept(self, output: Tensor) -> None: ...

    def decode(self, tokenizer: Tokenizer) -> str: ...

    def clear(self) -> None: ...
