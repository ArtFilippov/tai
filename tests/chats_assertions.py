"""Typed wire contracts and JSON shape assertions for chat integration tests."""

from collections import deque
from typing import Protocol, Required, TypedDict, cast

from http_peer import Reply


class RequestBody(TypedDict, total=False):
    model: Required[str]
    messages: Required[list[dict[str, str]]]
    metadata: dict[str, object]
    stop: list[str]
    temperature: float
    max_tokens: int
    seed: int
    n: int
    stream: bool


type RequestRecord = tuple[str, dict[str, str], RequestBody]


class HttpEndpoint(Protocol):
    requests: list[RequestRecord]
    scripts: dict[str, deque[Reply]]
    url: str
    require_session: bool


class EventRecord(TypedDict):
    kind: str
    data: dict[str, object]
    sequence: int
    run_id: str
    timestamp: str


def object_at(value: object, *path: str | int) -> dict[str, object]:
    value = at(value, *path)
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def list_at(value: object, *path: str | int) -> list[object]:
    value = at(value, *path)
    assert isinstance(value, list)
    return cast(list[object], value)


def text_at(value: object, *path: str | int) -> str:
    value = at(value, *path)
    assert isinstance(value, str)
    return value


def at(value: object, *path: str | int) -> object:
    for key in path:
        if isinstance(key, str):
            assert isinstance(value, dict)
            value = cast(dict[str, object], value)[key]
        else:
            assert isinstance(value, list)
            value = cast(list[object], value)[key]
    return value
