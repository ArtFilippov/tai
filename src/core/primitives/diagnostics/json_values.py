"""Recursive JSON values shared by diagnostic snapshots and persistence."""

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
