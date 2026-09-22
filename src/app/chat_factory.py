"""Compose transport adapters from validated, credential-aware configuration."""

import os
from collections.abc import Callable
from functools import partial

from core.chats import LLMChat
from core.chats.openai import OpenAIChat
from core.chats.qwen import QwenChat
from core.chats.qwen_server import QwenServer
from core.chats.server_chat import QwenServerChat

from .configuration_values import (
    OPENAI_BACKEND, QWEN_BACKEND, boolean, string, table, timeout_seconds,
)


def _credential(data: dict[str, object], role: str) -> tuple[str | None, str | None]:
    credentials = [key for key in ("api_key", "authorization", "api_key_env", "authorization_env") if key in data]
    if len(credentials) > 1:
        raise ValueError(f"{role}: выберите ровно один способ авторизации")
    if not credentials:
        return None, None
    key = credentials[0]
    credential = string(data[key], f"{role}.{key}")
    if key.endswith("_env"):
        credential = _environment(credential, f"{role}.{key}")
    return (credential, None) if key.startswith("api_key") else (None, credential)


def _environment(name: str, setting: str) -> str:
    if not name or name not in os.environ:
        raise ValueError(f"{setting}: переменная окружения не задана")
    return os.environ[name]


def _qwen_chat(value: object, role: str) -> Callable[[], LLMChat]:
    data = table(value, role, {"backend", "model", "revision", "cache_dir", "device", "dtype", "system", "generation"})
    return partial(QwenChat,
        model=string(data.get("model", ""), f"{role}.model"),
        revision=string(data.get("revision", ""), f"{role}.revision"),
        cache_dir=string(data.get("cache_dir", ""), f"{role}.cache_dir"),
        device=string(data.get("device", ""), f"{role}.device"),
        system=string(data.get("system", ""), f"{role}.system"),
        dtype=string(data.get("dtype", "float16"), f"{role}.dtype"),
        generation=table(data.get("generation", {}), f"{role}.generation"),
    )


def _openai_chat(value: object, role: str) -> Callable[[], LLMChat]:
    data = table(value, role, {
        "backend", "base_url", "model", "system", "generation", "timeout", "api_key",
        "authorization", "api_key_env", "authorization_env", "opencode_session",
    })
    api_key, authorization = _credential(data, role)
    return partial(OpenAIChat,
        base_url=string(data.get("base_url", ""), f"{role}.base_url"),
        model=string(data.get("model", ""), f"{role}.model"),
        system=string(data.get("system", ""), f"{role}.system"),
        api_key=api_key, authorization=authorization,
        timeout=timeout_seconds(data.get("timeout", 120.0), role),
        generation=table(data.get("generation", {}), f"{role}.generation"),
        opencode_session=boolean(data.get("opencode_session", False), f"{role}.opencode_session"),
    )


def _server_chat(value: object, role: str, server: QwenServer | None) -> Callable[[], LLMChat]:
    if server is None:
        raise ValueError(f"{role}: backend=vllm requires [server]")
    data = table(value, role, {"backend", "system", "generation", "timeout"})
    return partial(
        QwenServerChat, server, system=string(data.get("system", ""), f"{role}.system"),
        generation=table(data.get("generation", {}), f"{role}.generation"),
        timeout=timeout_seconds(data.get("timeout", 120.0), role),
    )


def chat_factory(value: object, role: str, server: QwenServer | None = None) -> Callable[[], LLMChat]:
    data = table(value, role)
    backend = data.get("backend", OPENAI_BACKEND)
    factories = {QWEN_BACKEND: _qwen_chat, OPENAI_BACKEND: _openai_chat,
                 "vllm": partial(_server_chat, server=server)}
    if not isinstance(backend, str) or backend not in factories:
        raise ValueError(f"{role}.backend: поддерживается openai, qwen или vllm")
    try:
        return factories[backend](value, role)
    except ValueError as exc:
        raise ValueError(f"{role}: {exc}") from None


def create_chat(value: object, role: str) -> LLMChat:
    return chat_factory(value, role)()
