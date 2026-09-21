"""JSON value validation shared by offline ALERT contracts."""

import json
from typing import cast

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type JsonObject = dict[str, Json]


def object_value(value: object, name: str, fields: set[str] | None = None) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected a JSON object")
    result = cast(dict[str, object], value)
    if fields is not None:
        missing, extra = fields - result.keys(), result.keys() - fields
        if missing or extra:
            raise ValueError(f"{name}: missing fields {sorted(missing)}; unknown fields {sorted(extra)}")
    return result


def array_value(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name}: expected a JSON array")
    return cast(list[object], value)


def text_value(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected nonempty text")
    try:
        _ = value.encode("utf-8")
    except UnicodeError:
        raise ValueError(f"{name}: expected valid Unicode text") from None
    return value


def integer_value(value: object, name: str, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name}: expected a nonnegative integer")
    if value < 0 or (maximum is not None and value > maximum):
        bound = "" if maximum is None else f" <= {maximum}"
        raise ValueError(f"{name}: expected a nonnegative integer{bound}")
    return value


def expect(value: object, expected: object, name: str) -> None:
    # JSON equality distinguishes true/1/1.0, unlike Python container equality.
    if json.dumps(value, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError(f"{name}: does not match the supported ALERT manifest contract")


def _unique_object(pairs: list[tuple[str, Json]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def decode_json(text: str) -> Json:
    return cast(Json, json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant))


def decode_json_lines(text: str) -> list[Json]:
    rows: list[Json] = []
    for index, line in enumerate(text.splitlines(), 1):
        try:
            rows.append(decode_json(line))
        except ValueError as exc:
            raise ValueError(f"Source line {index}: {exc}") from None
    return rows
