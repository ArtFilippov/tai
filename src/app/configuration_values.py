"""Validation of decoded experiment configuration values."""

from typing import cast

DEFAULT_ROUNDS = 10
MODEL_EVALUATOR = "model"
TEXT_EVALUATOR = "contains_text"
QWEN_BACKEND = "qwen"
OPENAI_BACKEND = "openai"


def table(value: object, name: str, allowed: set[str] | None = None) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name}: требуется TOML-таблица")
    data = cast(dict[str, object], value)
    if allowed is not None and data.keys() - allowed:
        raise ValueError(f"{name}: неизвестные параметры; проверьте конфигурацию")
    return data


def string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name}: требуется строка")
    return value


def boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name}: требуется boolean")
    return value


def timeout_seconds(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{role}.timeout: требуется число секунд")
    return float(value)


def round_limit(value: object) -> int:
    run = table(value, "run", {"generations", "max_calls_per_conv"})
    generations = run.get("generations", 1)
    if not isinstance(generations, int) or isinstance(generations, bool) or generations != 1:
        raise ValueError("run.generations: поддерживается только 1")
    max_calls = run.get("max_calls_per_conv", DEFAULT_ROUNDS)
    if not isinstance(max_calls, int) or isinstance(max_calls, bool) or max_calls <= 0:
        raise ValueError("run.max_calls_per_conv: требуется положительное целое")
    return max_calls
