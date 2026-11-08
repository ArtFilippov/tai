"""Deferred resources for a Transformers causal language model."""

from typing import Protocol

from transformers import AutoModelForCausalLM, AutoTokenizer


class TransformersAssets(Protocol):
    """Expose compatible weights and a chat tokenizer on first access."""

    @property
    def model(self) -> AutoModelForCausalLM:
        ...

    @property
    def tokenizer(self) -> AutoTokenizer:
        ...
