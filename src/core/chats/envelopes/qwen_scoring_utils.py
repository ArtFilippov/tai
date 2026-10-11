"""Pinned AutoDAN full-template logit loss on shared local Qwen weights."""

from importlib import import_module
from typing import Any

from ..qwen_context_error import QwenContextError
from ..qwen_diagnostic_utils import limits, scoring


def tokens(
    tokenizer: Any, messages: list[dict[str, str]], expected: str, device: str,
) -> tuple[Any, Any]:
    encode = getattr(tokenizer, "__call__")
    labels = encode([expected], add_special_tokens=False, return_tensors="pt")["input_ids"]
    target_ids = labels.to(device)
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt",
    ).to(device)
    return target_ids, inputs


def shift(model: Any, inputs: Any, target_ids: Any, tokenizer: Any) -> int:
    length, limit = limits(model, inputs, tokenizer)
    shift_value = length - target_ids.shape[-1]
    if length > limit or shift_value < 1:
        raise QwenContextError(scoring(model, inputs, tokenizer))
    return shift_value


def loss(model: Any, target_ids: Any, inputs: Any, tokenizer: Any) -> float:
    offset = shift(model, inputs, target_ids, tokenizer)
    forward = getattr(model, "__call__")
    logits = forward(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
    selected = logits[..., offset - 1:-1, :].contiguous()
    return mean(selected, target_ids)


def mean(selected: Any, target_ids: Any) -> float:
    flatten = getattr(selected, "view")(-1, getattr(selected, "size")(-1))
    labels = target_ids.flatten()
    cross_entropy = getattr(import_module("torch.nn.functional"), "cross_entropy")
    result = cross_entropy(flatten, labels, reduction="none").mean()
    return float(getattr(result, "item")())


def compute(
    libraries: tuple[Any, Any, Any], messages: list[dict[str, str]],
    expected: str, device: str,
) -> float:
    torch, tokenizer, model = libraries
    cuda = torch.cuda
    with torch.inference_mode(), getattr(cuda, "device")(device):
        try:
            return loss(model, *tokens(tokenizer, messages, expected, device), tokenizer)
        finally:
            getattr(cuda, "synchronize")(device)


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
    return compute(libraries, messages, expected, device)
