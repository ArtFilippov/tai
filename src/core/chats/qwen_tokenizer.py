"""The local chat-template and decoding boundary of Transformers."""

from typing import Protocol

from .qwen_input_batch import InputBatch
from .qwen_tensor import Tensor


class Tokenizer(Protocol):
    """Produce complete chat inputs and decode only new output tokens."""

    @property
    def model_max_length(self) -> int: ...

    @property
    def pad_token_id(self) -> int | None: ...

    def apply_chat_template(
        self, messages: list[dict[str, str]], **options: object,
    ) -> InputBatch: ...

    def decode(self, tokens: Tensor, **options: object) -> str: ...
