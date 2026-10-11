"""Validate explicit HTTP, credential and generation settings for one model role."""

import json
import math
from collections.abc import Mapping
from typing import cast
from urllib.parse import urlsplit

from core.primitives.diagnostics import DiagnosticRedactor, RedactionFactory


def require(valid: bool, message: str) -> None:
    if not valid:
        raise ValueError(message)


def validate_text(value: object, name: str, *, nonempty: bool = False) -> str:
    if isinstance(value, str):
        if nonempty:
            require(bool(value.strip()), f"{name} must be a nonempty string")
        return value
    raise ValueError(f"{name} must be a string")


def url_valid(value: str) -> bool:
    url = urlsplit(value)
    try:
        _ = url.port
    except ValueError:
        return False
    return (
        url.scheme in {"http", "https"} and bool(url.hostname)
        and (url.username, url.password) == (None, None)
        and not (url.query or url.fragment)
    )


def base_url(value: str) -> str:
    if url_valid(value) and not any(character.isspace() for character in value):
        return value.rstrip("/")
    raise ValueError(
        "base_url must be an HTTP(S) API root without credentials, query or fragment"
    )


def credential(value: object) -> str | None:
    if value is not None:
        text = validate_text(value, "authorization", nonempty=True)
        require(
            valid_header_characters(text),
            "authorization must be a nonempty single-line ASCII value",
        )
        return text
    return None


def valid_header_characters(value: str) -> bool:
    return value.isascii() and value.isprintable()


def parameters(value: object) -> dict[str, object]:
    if value is None:
        return {}
    require(isinstance(value, Mapping), "generation must contain finite JSON values")
    values: dict[str, object] = {}
    for key, entry in cast(Mapping[object, object], value).items():
        if isinstance(key, str):
            values[key] = entry
        else:
            raise ValueError("generation must use string keys")
    return values


def generation(value: object) -> str:
    values = parameters(value)
    require(
        not {"model", "messages"} & values.keys(), "generation cannot override model or messages",
    )
    if values.get("n", 1) != 1 or values.get("stream", False) is not False:
        raise ValueError("only n=1 and stream=False are supported")
    try:
        return json.dumps({**values, "n": 1, "stream": False}, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("generation must contain finite JSON values") from None


def session(value: object) -> bool:
    match value:
        case bool():
            return value
        case _:
            raise ValueError("opencode_session must be a boolean")


def deadline(value: object) -> float:
    match value:
        case int() | float():
            require(math.isfinite(value) and value > 0, "timeout must be positive and finite")
            return float(value)
        case _:
            raise ValueError("timeout must be positive and finite")


def authorization(
    options: Mapping[str, object], redaction: RedactionFactory,
) -> tuple[str | None, str, DiagnosticRedactor]:
    api_key = credential(options.get("api_key"))
    authorization_value = credential(options.get("authorization"))
    valid = api_key is None or authorization_value is None
    require(valid, "configure api_key or authorization, not both")
    values = (
        api_key, authorization_value,
        authorization_value.split(" ", 1)[-1] if authorization_value else None,
    )
    return (
        f"Bearer {api_key}" if api_key is not None else authorization_value,
        "api_key" if api_key is not None else (
            "explicit" if authorization_value is not None else "none"
        ),
        redaction(tuple(value for value in values if value)),
    )


def connection_config(
    url: str, options: Mapping[str, object], redaction: RedactionFactory,
) -> tuple[str, str | None, str, DiagnosticRedactor, str, float, bool]:
    if unknown := options.keys() - {
        "api_key", "authorization", "system", "generation", "timeout", "opencode_session",
    }:
        raise TypeError(f"Unexpected OpenAIChat options: {', '.join(sorted(unknown))}")
    _ = validate_text(options.get("system", ""), "system")
    header, mode, policy = authorization(options, redaction)
    return (
        base_url(url), header, mode, policy, generation(options.get("generation")),
        deadline(options.get("timeout", 120.0)), session(options.get("opencode_session", False)),
    )
