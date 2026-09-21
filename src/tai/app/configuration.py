"""Validate a complete experiment before starting any work or creating artifacts."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tai.attacks.fitd import FITDExperiment
from tai.core.chats import LLMChat, ModelRole
from tai.core.chats.openai import OpenAIChat
from tai.core.chats.qwen import QwenChat
from tai.core.evaluation.assembly import EvaluatorSetup
from tai.core.evaluation.model_evaluator import MODEL_RUBRIC_VERSION
from tai.core.primitives.diagnostics import DiagnosticRedactor
from tai.core.primitives.messages import Message


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
class RoleProfile:
    target: LLMChat
    attacker: LLMChat
    evaluator: EvaluatorSetup
    max_calls_per_conv: int

    def __post_init__(self) -> None:
        chats = tuple(self.chats().values())
        if len({id(chat) for chat in chats}) != len(chats):
            raise ValueError("Participants must use independent LLMChat instances")

    def chats(self) -> dict[ModelRole, LLMChat]:
        chats: dict[ModelRole, LLMChat] = {
            "Target": self.target,
            "Attacker": self.attacker,
        }
        if self.evaluator.chat is not None:
            chats["Evaluator"] = self.evaluator.chat
        return chats

    def redactor(self) -> DiagnosticRedactor:
        return DiagnosticRedactor.combine(
            *(chat._redactor for chat in self.chats().values())
        )

    def settings(self) -> dict[str, Any]:
        redactor = self.redactor()
        return {
            "evaluator": redactor.snapshot(self.evaluator.settings()),
            **{
                role: chat._diagnostic_settings(redactor)
                for role, chat in self.chats().items()
            },
        }

    def create_experiment(
        self, base: Message, journal_dir: Path | None = None
    ) -> FITDExperiment:
        return FITDExperiment(
            self.target,
            self.attacker,
            base,
            self.evaluator,
            journal_dir=journal_dir,
            max_calls_per_conv=self.max_calls_per_conv,
        )


def load_profile(config: Path, *, alert_series: bool = False) -> RoleProfile:
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
    if alert_series and evaluation.get("type") != "model":
        raise ValueError("ALERT series requires evaluator.type=model")
    evaluator: EvaluatorSetup
    if evaluation.get("type") == "contains_text":
        evaluation = _table(evaluation, "evaluator", {"type", "criterion"})
        evaluator = EvaluatorSetup.contains_text(
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
        evaluator = EvaluatorSetup.model(
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
    if alert_series and max_calls != 20:
        raise ValueError("ALERT series requires run.max_calls_per_conv=20")
    return RoleProfile(target, attacker, evaluator, max_calls)
