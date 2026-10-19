"""Validate server resource and HTTP generation parameters before startup."""

import math
from collections.abc import Mapping

from ..qwen_generation_utils import generation_config


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError(f"{name} must be a positive integer")


def positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def http_generation(parameters: Mapping[str, object] | None) -> dict[str, object]:
    values = generation_config(parameters)
    return {
        "max_tokens": values["max_new_tokens"], "temperature": values.get("temperature", 0.0),
        "top_p": values.get("top_p", 1.0), "top_k": values.get("top_k", 0) or -1,
        "repetition_penalty": values["repetition_penalty"], "n": 1, "stream": False,
        "seed": 0,
    }
