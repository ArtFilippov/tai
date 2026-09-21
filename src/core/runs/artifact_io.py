"""Atomic UTF-8 artifact writes shared by run, series, and report adapters."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


def render_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


@contextmanager
def _temporary(path: Path) -> Generator[Path]:
    temporary = path.with_name(path.name + ".tmp")
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def write_text(path: Path, text: str) -> None:
    """Atomically replace one file; propagate write, replace, and cleanup failures.

    Args:
        path (Path): Destination file in an existing directory.
        text (str): Exact UTF-8 text, without newline translation.
    """
    with _temporary(path) as temporary:
        _ = temporary.write_text(text, encoding="utf-8", newline="")
        _ = temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    """Atomically write formatted JSON, rejecting non-finite numeric values.

    Args:
        path (Path): Destination file in an existing directory.
        value (object): JSON-serializable artifact contents.
    """
    write_text(path, render_json(value))
