"""Local generation or scoring context overflow."""

from typing import override

from .qwen_error import QwenError
from .qwen_generation_buffers_contract import GenerationBuffersPort
from .qwen_model import Model
from .qwen_tokenizer import Tokenizer


class QwenContextError(QwenError):
    """The complete request would exceed the prepared model's context window."""

    @override
    def category(self) -> str:
        return "context"

    def enforce(
        self, buffers: GenerationBuffersPort, tokenizer: Tokenizer, model: Model,
        maximum: int | float | bool,
    ) -> None:
        """Reject a full request that would exceed the model context.

        Args:
            buffers (GenerationBuffersPort): Tokenized complete dialogue.
            tokenizer (Tokenizer): Tokenizer's context capacity.
            model (Model): Loaded model's position limit.
            maximum (int | float | bool): Requested maximum output length.

        Raises:
            QwenContextError: The complete request plus output exceeds the limit.
        """
        length = buffers.batch()["input_ids"].shape[-1]
        limit = min(model.config.max_position_embeddings, tokenizer.model_max_length)
        if length + maximum > limit:
            raise QwenContextError(
                f"Qwen {self.category()} overflow: input={length}, "
                + f"max_new_tokens={maximum}, "
                + f"limit={limit}; no messages were truncated"
            )
