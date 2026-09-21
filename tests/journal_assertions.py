"""Runtime narrowing for diagnostic objects, including collision-safe JSON unions."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from typing import TypeGuard

from core.primitives.messages import Message
from core.runs import Evaluation, User


def _dictionary(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _string_dictionary(value: object) -> TypeGuard[dict[str, object]]:
    return _dictionary(value) and all(isinstance(key, str) for key in value)


def mapping(value: object) -> dict[str, object]:
    assert _string_dictionary(value), value
    return value


def _sequence(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def sequence(value: object) -> list[object]:
    assert _sequence(value), value
    return value


def at(value: object, *path: str | int) -> object:
    for key in path:
        value = mapping(value)[key] if isinstance(key, str) else sequence(value)[key]
    return value


def text(value: object) -> str:
    assert isinstance(value, str), value
    return value


def result_of(user: User) -> Message:
    result = user.result()
    assert result is not None
    return result


def evaluation_of(user: User) -> Evaluation:
    evaluation = user.status().evaluation
    assert evaluation is not None
    return evaluation


def run_directory(user: User, root: Path) -> Path:
    run_id = user.status().run_id
    assert run_id is not None
    return root / run_id


@contextmanager
def released_worker(user: User, release: Event) -> Generator[None]:
    try:
        yield
    finally:
        release.set()
        assert user.wait(10)
