"""Pure failure descriptions shared by local generation and scoring."""

from typing import Any

from .qwen_checkpoint_contract import LocalCheckpointPort
from .qwen_load_buffers_contract import LoadBuffersPort


def limits(model: Any, inputs: Any, tokenizer: Any) -> tuple[int, int]:
    length = inputs["input_ids"].shape[-1]
    limit = min(model.config.max_position_embeddings, tokenizer.model_max_length)
    return length, limit


def scoring(model: Any, inputs: Any, tokenizer: Any) -> str:
    length, limit = limits(model, inputs, tokenizer)
    return f"AutoDAN scoring context overflow or empty prefix: input={length}, limit={limit}"


def dtype(buffers: LoadBuffersPort) -> str:
    return f"Model loaded with unexpected dtype: {buffers.model().dtype}"


def unavailable(device: str) -> str:
    return f"CUDA device {device} is unavailable; no CPU fallback"


def unsupported(device: str) -> str:
    return f"{device} does not support bfloat16"


def missing(checkpoint: LocalCheckpointPort) -> str:
    return f"Pinned local Qwen files are missing; prepare with: {checkpoint.preparation}"


def failed(checkpoint: LocalCheckpointPort) -> str:
    return f"Local Qwen load failed; prepare with: {checkpoint.preparation}"
