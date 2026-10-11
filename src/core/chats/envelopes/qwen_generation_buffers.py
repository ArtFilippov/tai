"""Keep generation tensors in one clearable call-local holder."""

from typing import override

from ..qwen_context_error import QwenContextError
from ..qwen_generation_buffers_contract import GenerationBuffersPort
from ..qwen_input_batch import InputBatch
from ..qwen_model import Model
from ..qwen_tensor import Tensor
from ..qwen_tokenizer import Tokenizer


class GenerationBuffers(GenerationBuffersPort):
    """Release input and output tensors even when a traceback retains frames."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[InputBatch] = []
        self.outputs: list[Tensor] = []

    @override
    def batch(self) -> InputBatch:
        return self.inputs[0]

    @override
    def prepare(
        self, tokenizer: Tokenizer, model: Model,
        turns: list[dict[str, str]], maximum: int | float | bool,
    ) -> None:
        self.inputs.append(tokenizer.apply_chat_template(
            turns, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, truncation=False,
        ))
        QwenContextError("Qwen context overflow").enforce(self, tokenizer, model, maximum)

    @override
    def move(self, device: str) -> None:
        self.inputs[0] = self.batch().to(device)

    @override
    def accept(self, output: Tensor) -> None:
        self.outputs.append(output)

    @override
    def decode(self, tokenizer: Tokenizer) -> str:
        return tokenizer.decode(
            self.outputs[0][0, self.batch()["input_ids"].shape[-1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )

    @override
    def clear(self) -> None:
        self.inputs.clear()
        self.outputs.clear()
