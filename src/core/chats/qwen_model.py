"""The local model operations needed for generation."""

from collections.abc import Sequence
from typing import Protocol, Self

from .qwen_generation_config import GenerationConfig
from .qwen_model_config import ModelConfig
from .qwen_stopping_criterion import StoppingCriterion
from .qwen_tensor import Tensor


class Model(Protocol):
    """A prepared model which generates from an entire tokenized chat."""

    @property
    def dtype(self) -> object: ...

    @property
    def config(self) -> ModelConfig: ...

    @property
    def generation_config(self) -> GenerationConfig: ...

    def to(self, device: str) -> Self: ...

    def eval(self) -> Self: ...

    def generate(
        self, *, generation_config: GenerationConfig,
        stopping_criteria: Sequence[StoppingCriterion], **inputs: Tensor,
    ) -> Tensor: ...
