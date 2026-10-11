"""Optional Transformers module operations used for local inference."""

from collections.abc import Callable, Sequence
from abc import abstractmethod
from typing import Protocol

from .qwen_generation_config_factory import GenerationConfigFactory
from .qwen_model import Model
from .qwen_stopping_criterion import StoppingCriterion
from .qwen_tokenizer import Tokenizer
from .qwen_torch import Torch


class TransformersPort(Protocol):
    """Load pinned local files and supply per-request decoding factories."""

    def torch(self) -> Torch: ...

    def ensure(self) -> None: ...

    def prepare(self) -> Torch: ...

    @property
    @abstractmethod
    def generation_config(self) -> GenerationConfigFactory: ...

    @property
    @abstractmethod
    def stopping_criteria(
        self,
    ) -> Callable[[list[StoppingCriterion]], Sequence[StoppingCriterion]]: ...

    def load_tokenizer(self, snapshot: str) -> Tokenizer: ...

    def load_model(self, snapshot: str, dtype: object) -> Model: ...
