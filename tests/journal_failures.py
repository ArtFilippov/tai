"""Poisoned journal adapters for storage and cleanup boundary tests."""

from collections.abc import Callable
from io import TextIOBase
from pathlib import Path
from typing import IO, Protocol, Unpack

import pytest
from journal_assertions import mapping

from core.runs.journal import RunEvent, RunJournal
from core.runs.journal_configuration import JournalOptions


class JournalFactory(Protocol):
    def __call__(self, **options: Unpack[JournalOptions]) -> RunJournal: ...


def failing_journal_factory(
    monkeypatch: pytest.MonkeyPatch, failure: Exception,
    should_fail: Callable[[RunEvent], bool], attempted: list[RunEvent] | None = None,
) -> JournalFactory:
    def create(**options: Unpack[JournalOptions]) -> RunJournal:
        journal = RunJournal(**options)
        write_event = journal._write_event

        def write(event: RunEvent) -> None:
            if attempted is not None:
                attempted.append(event)
            if should_fail(event):
                raise failure
            write_event(event)

        monkeypatch.setattr(journal, "_write_event", write)
        return journal

    return create


class ClosingFailureStream:
    def __init__(self, source: TextIOBase) -> None:
        super().__init__()
        self.source = source

    def write(self, text: str) -> int:
        return self.source.write(text)

    def flush(self) -> None:
        self.source.flush()

    def close(self) -> None:
        self.source.close()
        raise RuntimeError("secondary close error")


def wrap_stream(
    path: Path, source: IO[str] | IO[bytes], streams: list[ClosingFailureStream],
) -> IO[str] | IO[bytes] | ClosingFailureStream:
    if path.name == "events.jsonl":
        assert isinstance(source, TextIOBase)
        streams.append(ClosingFailureStream(source))
        return streams[-1]
    return source


def broken_journal(
    monkeypatch: pytest.MonkeyPatch, kind: str, role: str | None, failure: Exception,
) -> tuple[JournalFactory, list[RunEvent], list[ClosingFailureStream]]:
    attempted: list[RunEvent] = []
    streams: list[ClosingFailureStream] = []
    open_file = Path.open

    def open_stream(
        path: Path, mode: str = "r", buffering: int = -1, encoding: str | None = None,
        errors: str | None = None, newline: str | None = None,
    ) -> IO[str] | IO[bytes] | ClosingFailureStream:
        return wrap_stream(path, open_file(path, mode, buffering, encoding, errors, newline), streams)

    monkeypatch.setattr(Path, "open", open_stream)
    factory = failing_journal_factory(
        monkeypatch, failure,
        lambda event: event.kind == kind and (role is None or mapping(event.data).get("role") == role),
        attempted,
    )
    return factory, attempted, streams
