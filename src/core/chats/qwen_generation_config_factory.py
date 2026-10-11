"""One generation configuration per local request."""

from abc import abstractmethod
from typing import Protocol

from .qwen_generation_config import GenerationConfig


class GenerationConfigFactory(Protocol):
    """Build model-compatible decoding options for a complete input."""

    @abstractmethod
    def __call__(self, **options: object) -> GenerationConfig: ...
