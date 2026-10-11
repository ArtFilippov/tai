"""Call-local holders for a partially loaded offline model."""

from threading import Event
from types import TracebackType
from typing import Protocol, Self

from .qwen_model import Model
from .qwen_tokenizer import Tokenizer
from .qwen_torch import Torch
from .qwen_transformers_contract import TransformersPort


class LoadBuffersPort(Protocol):
    """Release intermediate references before a failed load escapes."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def tokenizer(self) -> Tokenizer: ...

    def model(self) -> Model: ...

    def retain(self) -> None: ...

    def load_tokenizer(self, transformers: TransformersPort, snapshot: str) -> None: ...

    def load_model(self, transformers: TransformersPort, snapshot: str, dtype: object) -> None: ...

    def prepare(self, torch: Torch, device: str, dtype: object, cancellation: Event) -> None: ...

    def clear(self) -> None: ...
