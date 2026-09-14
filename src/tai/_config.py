"""Validate a complete experiment before starting any work or creating artifacts."""

import os
import tomllib
from pathlib import Path
from typing import Any

from tai import ContainsTextEvaluator, FITDUser, Message, OpenAIChat


def _table(value: Any, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name}: требуется TOML-таблица")
    if value.keys() - allowed:
        raise ValueError(f"{name}: неизвестные параметры; проверьте конфигурацию")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name}: требуется строка")
    return value


def _chat(value: Any, role: str) -> OpenAIChat:
    data = _table(
        value,
        role,
        {
            "base_url",
            "model",
            "system",
            "generation",
            "timeout",
            "api_key",
            "authorization",
            "api_key_env",
            "authorization_env",
        },
    )
    kwargs = {
        key: _string(data.get(key, ""), f"{role}.{key}")
        for key in ("base_url", "model", "system")
    }
    credentials = [
        key
        for key in ("api_key", "authorization", "api_key_env", "authorization_env")
        if key in data
    ]
    if len(credentials) > 1:
        raise ValueError(f"{role}: выберите ровно один способ авторизации")
    for key in credentials:
        credential = _string(data[key], f"{role}.{key}")
        if key.endswith("_env"):
            if not credential or credential not in os.environ:
                raise ValueError(f"{role}.{key}: переменная окружения не задана")
            credential = os.environ[credential]
        kwargs[key.removesuffix("_env")] = credential
    timeout = data.get("timeout", 120.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError(f"{role}.timeout: требуется число секунд")
    generation = data.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError(f"{role}.generation: требуется TOML-таблица")
    try:
        return OpenAIChat(**kwargs, timeout=timeout, generation=generation)
    except ValueError as exc:
        raise ValueError(f"{role}: {exc}") from None


def load_user(config: Path, base: Message, artifacts_dir: Path) -> FITDUser:
    with config.open("rb") as source:
        try:
            document = tomllib.load(source)
        except tomllib.TOMLDecodeError:
            # TOML errors can contain source snippets, including credentials.
            raise ValueError(
                "Некорректный TOML; проверьте синтаксис конфигурации"
            ) from None
    data = _table(document, "config", {"target", "attacker", "evaluator", "run"})
    target = _chat(data.get("target"), "target")
    attacker = _chat(data.get("attacker"), "attacker")
    evaluation = _table(data.get("evaluator"), "evaluator", {"type", "criterion"})
    if evaluation.get("type") != "contains_text":
        raise ValueError("evaluator.type: поддерживается contains_text")
    evaluator = ContainsTextEvaluator(
        _string(evaluation.get("criterion"), "evaluator.criterion")
    )
    run = _table(data.get("run", {}), "run", {"generations", "max_calls_per_conv"})
    generations = run.get("generations", 1)
    if type(generations) is not int or generations != 1:
        raise ValueError("run.generations: поддерживается только 1")
    return FITDUser(
        target,
        attacker,
        base,
        evaluator,
        artifacts_dir=artifacts_dir,
        max_calls_per_conv=run.get("max_calls_per_conv", 10),
    )
