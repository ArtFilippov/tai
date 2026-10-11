"""Validate Qwen decoding options without loading optional libraries."""

import math
from collections.abc import Mapping
from typing import cast


def integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def positive(value: object, name: str, maximum: float = math.inf) -> float | int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"{name} must be positive and finite (top_p <= 1)")
        return value
    raise ValueError(f"{name} must be positive and finite (top_p <= 1)")


def validated_parameters(parameters: object) -> dict[str, object]:
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise ValueError("generation must be a mapping")
    values: dict[str, object] = {}
    for key in cast(Mapping[object, object], parameters):
        if not isinstance(key, str):
            raise ValueError("generation parameter names must be strings")
        values[key] = parameters[key]
    allowed = {"temperature", "top_p", "top_k", "max_new_tokens", "do_sample", "repetition_penalty"}
    if values.keys() <= allowed:
        return values
    raise ValueError("unsupported Qwen generation parameters; see README schema")


def sampling(values: Mapping[str, object]) -> bool:
    value = values.get("do_sample", False)
    if isinstance(value, bool):
        if value or not {"temperature", "top_p", "top_k"} & values.keys():
            return value
        raise ValueError("temperature/top_p/top_k require do_sample=true")
    raise ValueError("do_sample must be a boolean")


def generation_config(parameters: object) -> Mapping[str, int | float | bool]:
    values = validated_parameters(parameters)
    sample = sampling(values)
    result: dict[str, int | float | bool] = {
        "max_new_tokens": integer(values.get("max_new_tokens", 512), "max_new_tokens", 1),
        "do_sample": sample,
        "repetition_penalty": positive(values.get("repetition_penalty", 1.0), "repetition_penalty"),
        "num_beams": 1, "num_return_sequences": 1,
        "use_cache": True, "return_dict_in_generate": False,
    }
    if sample:
        result.update({
            "temperature": positive(values.get("temperature", 1.0), "temperature"),
            "top_p": positive(values.get("top_p", 1.0), "top_p", 1),
            "top_k": integer(values.get("top_k", 50), "top_k", 0),
        })
    return result


def validate_dtype(dtype: object) -> str:
    match dtype:
        case str() as selected if selected in {"float16", "bfloat16", "float32"}:
            return selected
        case _:
            raise ValueError("dtype must be float16, bfloat16, or float32")
