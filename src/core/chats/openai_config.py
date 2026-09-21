"""Validated, immutable connection configuration."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict
from urllib.parse import urlsplit

from core.primitives.diagnostics import DiagnosticRedactor


class OpenAIOptions(TypedDict, total=False):
    api_key: str | None
    authorization: str | None
    system: str
    generation: Mapping[str, object] | None
    timeout: float
    opencode_session: bool


def validate_text(value: object, name: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if nonempty and not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _url_valid(base_url: str) -> bool:
    url = urlsplit(base_url)
    _ = url.port
    return (
        url.scheme in {"http", "https"} and bool(url.hostname)
        and (url.username, url.password) == (None, None)
        and not (url.query or url.fragment)
    )


def _base_url(base_url: str) -> str:
    try:
        valid = _url_valid(base_url)
    except ValueError:
        valid = False
    if not valid or any(character.isspace() for character in base_url):
        raise ValueError(
            "base_url must be an HTTP(S) API root without credentials, query or fragment"
        )
    return base_url.rstrip("/")


def _credential(value: object) -> str | None:
    if value is None:
        return None
    text = validate_text(value, "authorization", nonempty=True)
    if not _valid_header_characters(text):
        raise ValueError("authorization must be a nonempty single-line ASCII value")
    return text


def _valid_header_characters(value: str) -> bool:
    return value.isascii() and value.isprintable()


def _generation(parameters: Mapping[str, object] | None) -> str:
    values = dict(parameters or {})
    if {"model", "messages"} & values.keys():
        raise ValueError("generation cannot override model or messages")
    if values.get("n", 1) != 1 or values.get("stream", False) is not False:
        raise ValueError("only n=1 and stream=False are supported")
    try:
        return json.dumps({**values, "n": 1, "stream": False}, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("generation must contain finite JSON values") from None


def _session(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("opencode_session must be a boolean")
    return value


@dataclass(frozen=True)
class Authorization:
    header: str | None
    mode: str
    redactor: DiagnosticRedactor


def _authorization(options: OpenAIOptions) -> Authorization:
    api_key = _credential(options.get("api_key"))
    authorization = _credential(options.get("authorization"))
    if api_key is not None and authorization is not None:
        raise ValueError("configure api_key or authorization, not both")
    values = (api_key, authorization, authorization.split(" ", 1)[-1] if authorization else None)
    return Authorization(
        f"Bearer {api_key}" if api_key is not None else authorization,
        "api_key" if api_key is not None else "explicit" if authorization is not None else "none",
        DiagnosticRedactor(value for value in values if value),
    )


@dataclass(frozen=True)
class OpenAIConfig:
    base_url: str
    authorization: Authorization
    generation: str
    timeout: float
    session: bool


def connection_config(base_url: str, options: OpenAIOptions) -> OpenAIConfig:
    if unknown := options.keys() - {
        "api_key", "authorization", "system", "generation", "timeout", "opencode_session",
    }:
        raise TypeError(f"Unexpected OpenAIChat options: {', '.join(sorted(unknown))}")
    timeout = options.get("timeout", 120.0)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    return OpenAIConfig(
        _base_url(base_url), _authorization(options),
        _generation(options.get("generation")), timeout,
        _session(options.get("opencode_session", False)),
    )
