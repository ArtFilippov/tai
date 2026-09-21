"""Strict decoding of the model assessment's single boolean field."""

import json
from typing import Never

from core.primitives.diagnostics.json_values import JSONValue


def _object_pairs(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    value: dict[str, JSONValue] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def _invalid_constant(value: str) -> Never:
    raise ValueError(f"non-JSON constant: {value}")


def verdict(text: str) -> bool:
    value: JSONValue = json.loads(
        text, object_pairs_hook=_object_pairs, parse_constant=_invalid_constant
    )
    if not isinstance(value, dict) or value.keys() != {"success"}:
        raise ValueError("invalid verdict schema")
    success = value["success"]
    if not isinstance(success, bool):
        raise ValueError("invalid verdict schema")
    return success
