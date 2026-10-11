"""Hold partially loaded weights and tokenizer until publication or failure."""

from threading import Event
from types import TracebackType
from typing import Self, override

from ..qwen_failure import QwenFailure
from ..qwen_diagnostic_utils import dtype as dtype_error
from ..qwen_load_buffers_contract import LoadBuffersPort
from ..qwen_model import Model
from ..qwen_tokenizer import Tokenizer
from ..qwen_torch import Torch
from ..qwen_transformers_contract import TransformersPort
from .qwen_gate_utils import check_cancelled


class LoadBuffers(LoadBuffersPort):
    """Keep partial model references in one clearable load-local holder."""

    def __init__(self) -> None:
        super().__init__()
        self.weights: tuple[list[Tokenizer], list[Model]] = ([], [])
        self.published = False

    @override
    def __enter__(self) -> Self:
        _ = super().__enter__()
        return self

    @override
    def __exit__(
        self, kind: type[BaseException] | None, exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.clear()
        super().__exit__(kind, exception, traceback)

    @override
    def tokenizer(self) -> Tokenizer:
        return self.weights[0][0]

    @override
    def model(self) -> Model:
        return self.weights[1][0]

    @override
    def retain(self) -> None:
        self.published = True

    @override
    def load_tokenizer(self, transformers: TransformersPort, snapshot: str) -> None:
        self.weights[0].append(transformers.load_tokenizer(snapshot))

    @override
    def load_model(self, transformers: TransformersPort, snapshot: str, dtype: object) -> None:
        self.weights[1].append(transformers.load_model(snapshot, dtype))

    @override
    def prepare(self, torch: Torch, device: str, dtype: object, cancellation: Event) -> None:
        check_cancelled(cancellation)
        self.weights[1][0] = self.model().to(device).eval()
        torch.cuda.synchronize(device)
        check_cancelled(cancellation)
        if self.model().dtype != dtype:
            raise QwenFailure(dtype_error(self))

    @override
    def clear(self) -> None:
        if not self.published:
            self.weights[0].clear()
            self.weights[1].clear()
