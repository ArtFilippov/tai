"""Offline Transformers loading behind a narrow local inference port."""

from collections.abc import Callable, Sequence
from typing import cast, override

from .qwen_generation_config_factory import GenerationConfigFactory
from .qwen_import_contract import ImportPort
from .qwen_model import Model
from .qwen_model_loader import ModelLoader
from .qwen_stopping_criterion import StoppingCriterion
from .qwen_tokenizer import Tokenizer
from .qwen_tokenizer_loader import TokenizerLoader
from .qwen_torch import Torch
from .qwen_transformers_contract import TransformersPort


class Transformers(TransformersPort):
    """A lazily imported module with fixed local-only loading policy."""

    def __init__(self, importer: ImportPort) -> None:
        """Hold the optional import boundary without loading model weights.

        Args:
            importer (ImportPort): Lazy optional-library import.
        """
        super().__init__()
        self.importer: ImportPort = importer

    @override
    def torch(self) -> Torch:
        return cast(Torch, self.module("torch"))

    @override
    def ensure(self) -> None:
        _ = self.module()

    @override
    def prepare(self) -> Torch:
        torch = self.torch()
        self.ensure()
        return torch

    def module(self, name: str = "transformers") -> object:
        return self.importer(name)

    @property
    @override
    def generation_config(self) -> GenerationConfigFactory:
        return cast(GenerationConfigFactory, getattr(self.module(), "GenerationConfig"))

    @property
    @override
    def stopping_criteria(self) -> Callable[[list[StoppingCriterion]], Sequence[StoppingCriterion]]:
        return cast(
            Callable[[list[StoppingCriterion]], Sequence[StoppingCriterion]],
            getattr(self.module(), "StoppingCriteriaList"),
        )

    @override
    def load_tokenizer(self, snapshot: str) -> Tokenizer:
        return cast(TokenizerLoader, getattr(self.module(), "AutoTokenizer")).from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False,
        )

    @override
    def load_model(self, snapshot: str, dtype: object) -> Model:
        return cast(ModelLoader, getattr(self.module(), "AutoModelForCausalLM")).from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=dtype, attn_implementation="sdpa",
        )
