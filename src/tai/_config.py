"""Validate a complete experiment before starting any work or creating artifacts."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tai import (
    ContainsTextEvaluator,
    Evaluator,
    FITDExperiment,
    LLMChat,
    Message,
    ModelEvaluator,
    OpenAIChat,
    QwenChat,
)
from tai._diagnostics import DiagnosticRedactor
from tai.evaluators import MODEL_RUBRIC_VERSION


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


def _chat(value: Any, role: str) -> LLMChat:
    if not isinstance(value, dict):
        raise ValueError(f"{role}: требуется TOML-таблица")
    backend = value.get("backend", "openai")
    if backend == "qwen":
        data = _table(
            value,
            role,
            {
                "backend",
                "model",
                "revision",
                "cache_dir",
                "device",
                "dtype",
                "system",
                "generation",
            },
        )
        kwargs = {
            key: _string(data.get(key, ""), f"{role}.{key}")
            for key in ("model", "revision", "cache_dir", "device", "system")
        }
        generation = data.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError(f"{role}.generation: требуется TOML-таблица")
        try:
            return QwenChat(
                **kwargs,
                dtype=data.get("dtype", "float16"),
                generation=generation,
            )
        except ValueError as exc:
            raise ValueError(f"{role}: {exc}") from None
    if backend != "openai":
        raise ValueError(f"{role}.backend: поддерживается openai или qwen")
    return _openai_chat(value, role)


def _openai_chat(value: Any, role: str) -> OpenAIChat:
    data = _table(
        value,
        role,
        {
            "backend",
            "base_url",
            "model",
            "system",
            "generation",
            "timeout",
            "api_key",
            "authorization",
            "api_key_env",
            "authorization_env",
            "opencode_session",
        },
    )
    kwargs: dict[str, Any] = {
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
        return OpenAIChat(
            **kwargs,
            timeout=timeout,
            generation=generation,
            opencode_session=data.get("opencode_session", False),
        )
    except ValueError as exc:
        raise ValueError(f"{role}: {exc}") from None


@dataclass(frozen=True)
class ModelEvaluatorProfile:
    """Unbound classifier configuration; each base gets its own Evaluator."""

    chat: LLMChat
    rubric_version: str


@dataclass(frozen=True)
class RoleProfile:
    target: LLMChat
    attacker: LLMChat
    evaluator: ContainsTextEvaluator | ModelEvaluatorProfile
    max_calls_per_conv: int

    def redactor(self) -> DiagnosticRedactor:
        chats = [self.target, self.attacker]
        if isinstance(self.evaluator, ModelEvaluatorProfile):
            chats.append(self.evaluator.chat)
        return DiagnosticRedactor.combine(*(chat._redactor for chat in chats))

    def create_evaluator(self, base: Message) -> Evaluator:
        if isinstance(self.evaluator, ModelEvaluatorProfile):
            return ModelEvaluator(
                self.evaluator.chat, base, self.evaluator.rubric_version
            )
        return self.evaluator

    def create_experiment(self, base: Message, artifacts_dir: Path) -> FITDExperiment:
        return FITDExperiment(
            self.target,
            self.attacker,
            base,
            self.create_evaluator(base),
            artifacts_dir=artifacts_dir,
            max_calls_per_conv=self.max_calls_per_conv,
        )


def load_profile(config: Path) -> RoleProfile:
    """One validated profile shared by readiness and terminal experiments."""
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
    evaluation = data.get("evaluator")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluator: требуется TOML-таблица")
    evaluator: ContainsTextEvaluator | ModelEvaluatorProfile
    if evaluation.get("type") == "contains_text":
        evaluation = _table(evaluation, "evaluator", {"type", "criterion"})
        evaluator = ContainsTextEvaluator(
            _string(evaluation.get("criterion"), "evaluator.criterion")
        )
    elif evaluation.get("type") == "model":
        evaluation = _table(evaluation, "evaluator", {"type", "rubric_version", "chat"})
        rubric_version = _string(
            evaluation.get("rubric_version"), "evaluator.rubric_version"
        )
        if rubric_version != MODEL_RUBRIC_VERSION:
            raise ValueError(
                f"evaluator.rubric_version: требуется {MODEL_RUBRIC_VERSION}"
            )
        evaluator = ModelEvaluatorProfile(
            _chat(evaluation.get("chat"), "evaluator.chat"), rubric_version
        )
    else:
        raise ValueError("evaluator.type: поддерживается contains_text или model")
    run = _table(data.get("run", {}), "run", {"generations", "max_calls_per_conv"})
    generations = run.get("generations", 1)
    if type(generations) is not int or generations != 1:
        raise ValueError("run.generations: поддерживается только 1")
    max_calls = run.get("max_calls_per_conv", 10)
    if type(max_calls) is not int or max_calls <= 0:
        raise ValueError("run.max_calls_per_conv: требуется положительное целое")
    return RoleProfile(target, attacker, evaluator, max_calls)
