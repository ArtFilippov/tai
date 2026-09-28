"""Pinned AutoDAN full-template logit loss on shared local Qwen weights."""

from contextlib import contextmanager
from importlib import import_module
from collections.abc import Generator
from typing import Any

from .qwen_config import QwenContextError


def _tokens(tokenizer: Any, messages: list[dict[str, str]], expected: str, device: str) -> tuple[Any, Any]:
    encode = getattr(tokenizer, "__call__")
    labels = encode([expected], add_special_tokens=False, return_tensors="pt")["input_ids"]
    target_ids = labels.to(device)
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    ).to(device)
    return target_ids, inputs


def _shift(model: Any, inputs: Any, target_ids: Any, tokenizer: Any) -> int:
    length = inputs["input_ids"].shape[-1]
    limit = min(model.config.max_position_embeddings, tokenizer.model_max_length)
    shift = length - target_ids.shape[-1]
    if length > limit or shift < 1:
        raise QwenContextError(f"AutoDAN scoring context overflow or empty prefix: input={length}, limit={limit}")
    return shift


def _loss(model: Any, inputs: Any, target_ids: Any, tokenizer: Any) -> float:
    shift = _shift(model, inputs, target_ids, tokenizer)
    forward = getattr(model, "__call__")
    logits = forward(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
    selected = logits[..., shift - 1:-1, :].contiguous()
    return _mean_loss(selected, target_ids)


def _mean_loss(selected: Any, target_ids: Any) -> float:
    flatten = getattr(selected, "view")(-1, getattr(selected, "size")(-1))
    labels = target_ids.view(-1)
    cross_entropy = getattr(import_module("torch.nn.functional"), "cross_entropy")
    loss = cross_entropy(flatten, labels, reduction="none").mean()
    return float(getattr(loss, "item")())


@contextmanager
def _device(torch: Any, device: str) -> Generator[None]:
    cuda = torch.cuda
    with torch.inference_mode(), getattr(cuda, "device")(device):
        try:
            yield
        finally:
            getattr(cuda, "synchronize")(device)


def _compute(libraries: tuple[Any, Any, Any], messages: list[dict[str, str]], expected: str, device: str) -> float:
    torch, tokenizer, model = libraries
    with _device(torch, device):
        target_ids, inputs = _tokens(tokenizer, messages, expected, device)
        return _loss(model, inputs, target_ids, tokenizer)


def score_continuation(
    libraries: tuple[Any, Any, Any], prompt: str, expected: str, system: str, *, device: str,
) -> float:
    """Apply pinned garak token alignment without conversation state.

    Args:
        libraries (tuple[Any, Any, Any]): Torch, shared tokenizer and shared model.
        prompt (str): Full user prompt.
        expected (str): Expected assistant continuation.
        system (str): Target system instruction.
        device (str): Pinned CUDA device.

    Returns:
        float: Mean token cross-entropy.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}, {"role": "assistant", "content": expected},
    ]
    return _compute(libraries, messages, expected, device)
