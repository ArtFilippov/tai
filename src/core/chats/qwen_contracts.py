"""Structural contracts for the optional Torch and Transformers runtime boundary."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from threading import Event
from typing import Protocol, Self, cast

from .chat_turn_contract import ChatTurn
from .qwen_config import Generation, LocalCheckpoint


class Tensor(Protocol):
    @property
    @abstractmethod
    def shape(self) -> Sequence[int]: ...

    @abstractmethod
    def __getitem__(self, index: tuple[int, slice]) -> Tensor: ...


class InputBatch(Protocol):
    @abstractmethod
    def keys(self) -> Iterable[str]: ...

    @abstractmethod
    def __getitem__(self, key: str) -> Tensor: ...

    @abstractmethod
    def to(self, device: str) -> Self: ...


class Tokenizer(Protocol):
    @property
    @abstractmethod
    def model_max_length(self) -> int: ...

    @property
    @abstractmethod
    def pad_token_id(self) -> int | None: ...

    @abstractmethod
    def apply_chat_template(
        self, messages: list[dict[str, str]], **options: object,
    ) -> InputBatch: ...

    @abstractmethod
    def decode(self, tokens: Tensor, **options: object) -> str: ...


class ModelConfig(Protocol):
    @property
    @abstractmethod
    def max_position_embeddings(self) -> int: ...


class GenerationConfig(Protocol):
    @property
    @abstractmethod
    def bos_token_id(self) -> int | None: ...

    @property
    @abstractmethod
    def eos_token_id(self) -> int | list[int] | None: ...


class StoppingCriterion(Protocol):
    @abstractmethod
    def __call__(self, *args: object, **kwargs: object) -> bool: ...


class Model(Protocol):
    @property
    @abstractmethod
    def dtype(self) -> object: ...

    @property
    @abstractmethod
    def config(self) -> ModelConfig: ...

    @property
    @abstractmethod
    def generation_config(self) -> GenerationConfig: ...

    @abstractmethod
    def to(self, device: str) -> Self: ...

    @abstractmethod
    def eval(self) -> Self: ...

    @abstractmethod
    def generate(
        self, *, generation_config: GenerationConfig,
        stopping_criteria: Sequence[StoppingCriterion], **inputs: Tensor,
    ) -> Tensor: ...


class Cuda(Protocol):
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def device_count(self) -> int: ...

    @abstractmethod
    def is_bf16_supported(self) -> bool: ...

    @abstractmethod
    def device(self, device: str) -> AbstractContextManager[object]: ...

    @abstractmethod
    def synchronize(self, device: str) -> None: ...


class Torch(Protocol):
    @property
    @abstractmethod
    def cuda(self) -> Cuda: ...

    @property
    @abstractmethod
    def float16(self) -> object: ...

    @property
    @abstractmethod
    def bfloat16(self) -> object: ...

    @property
    @abstractmethod
    def float32(self) -> object: ...

    @abstractmethod
    def inference_mode(self) -> AbstractContextManager[object]: ...


class TokenizerLoader(Protocol):
    @abstractmethod
    def from_pretrained(self, path: str, **options: object) -> Tokenizer: ...


class ModelLoader(Protocol):
    @abstractmethod
    def from_pretrained(self, path: str, **options: object) -> Model: ...


class GenerationConfigFactory(Protocol):
    @abstractmethod
    def __call__(self, **options: object) -> GenerationConfig: ...


@dataclass(frozen=True)
class Transformers:
    tokenizer_loader: TokenizerLoader
    model_loader: ModelLoader
    generation_config: GenerationConfigFactory
    stopping_criteria: Callable[[list[StoppingCriterion]], Sequence[StoppingCriterion]]

    def load_tokenizer(self, snapshot: str) -> Tokenizer:
        return self.tokenizer_loader.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False,
        )

    def load_model(self, snapshot: str, dtype: object) -> Model:
        return self.model_loader.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=dtype, attn_implementation="sdpa",
        )

    @classmethod
    def from_module(cls, module: object) -> Transformers:
        return cls(
            cast(TokenizerLoader, getattr(module, "AutoTokenizer")),
            cast(ModelLoader, getattr(module, "AutoModelForCausalLM")),
            cast(GenerationConfigFactory, getattr(module, "GenerationConfig")),
            cast(
                Callable[[list[StoppingCriterion]], Sequence[StoppingCriterion]],
                getattr(module, "StoppingCriteriaList"),
            ),
        )


class LocalRuntimeProtocol(Protocol):
    @abstractmethod
    def score_continuation(self, prompt: str, expected: str, system: str, *, device: str) -> float: ...

    @abstractmethod
    def actual_dtype(self) -> str: ...

    @abstractmethod
    def generate(
        self, turns: tuple[ChatTurn, ...], cancellation: Event,
        generation: Generation, *, device: str,
    ) -> str: ...


class QwenExecutorProtocol(Protocol):
    device: str
    dtype: str

    @abstractmethod
    def actual_dtype(self) -> str | None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def score_continuation(
        self, prompt: str, expected: str, system: str, *, checkpoint: LocalCheckpoint,
        importer: Callable[[str], object],
    ) -> float: ...

    @abstractmethod
    def generate(
        self, turns: tuple[ChatTurn, ...], cancellation: Event,
        generation: Generation, *, checkpoint: LocalCheckpoint,
        importer: Callable[[str], object],
    ) -> str: ...
