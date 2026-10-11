"""Validate explicit local checkpoint identities and cache paths."""

import re
from pathlib import Path


def identifier(value: object, pattern: str, reason: str) -> str:
    return text(value, pattern, reason)


def text(value: object, pattern: str | None, reason: str) -> str:
    if isinstance(value, str) and (pattern is None or re.fullmatch(pattern, value)):
        return value
    raise ValueError(reason)


def validate_system(system: object) -> str:
    return text(system, None, "system must be a string")


def cache(value: object) -> Path:
    if isinstance(value, (str, Path)) and str(value).strip():
        return Path(value)
    raise ValueError("cache_dir must name a prepared local Hugging Face cache")
