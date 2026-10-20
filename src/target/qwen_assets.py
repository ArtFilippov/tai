"""Read-only interface to a pinned local Qwen snapshot and its loaded resources."""

from typing import Protocol

from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenAssets(Protocol):
    """Supply one tokenizer and one set of weights from a local snapshot."""

    @property
    def path(self) -> str:
        ...

    @property
    def tokenizer(self) -> AutoTokenizer:
        ...

    @property
    def model(self) -> AutoModelForCausalLM:
        ...
