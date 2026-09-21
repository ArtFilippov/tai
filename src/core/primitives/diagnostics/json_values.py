"""Recursive JSON values shared by diagnostic snapshots and persistence."""

import json

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
type DiagnosticObject = dict[str, JSONValue] | list[JSONValue]


def json_value(text: str) -> JSONValue:
    return json.loads(text)


def json_object(text: str) -> DiagnosticObject:
    value = json_value(text)
    if not isinstance(value, (dict, list)):
        raise ValueError("Expected a JSON object or diagnostic key/value pairs")
    return value
