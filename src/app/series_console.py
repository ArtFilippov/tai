"""One stderr Live display, updated by addressed journals rather than event polling."""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from types import TracebackType
from typing import Self, TextIO, override

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from core.runs.console_ports import ConsoleRow, RowChange

from .console_format import ConsoleEntry, row_table, safe_text


@dataclass(frozen=True)
class RowAddress(ConsoleRow):
    deliver: Callable[[RowChange], None]

    @override
    def update(self, change: RowChange) -> None:
        self.deliver(change)


class SeriesConsole:
    def __init__(self, total: int, *, server: Callable[[], dict[str, object]] = dict, stream: TextIO | None = None) -> None:
        super().__init__()
        self._total = total
        self._server = server
        self._console = Console(file=stream or sys.stderr, markup=False, highlight=False)
        self._entries: dict[int, ConsoleEntry] = {}
        self._lock = RLock()
        self._live = Live(console=self._console, get_renderable=self.render, refresh_per_second=10,
                          redirect_stdout=False, redirect_stderr=False, vertical_overflow="crop")
        self._write = self._refresh if self._console.is_terminal else self._transition

    def __enter__(self) -> Self:
        if self._console.is_terminal:
            self._live.start(refresh=True)
        return self

    def __exit__(
        self, _kind: type[BaseException] | None, _value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._console.is_terminal:
            self._live.stop()
        else:
            self._console.print(self.render())

    def row(self, sequence: int, category: str) -> ConsoleRow:
        # Allocation is idle; only bind(run_id) makes it an admitted row.
        with self._lock:
            _ = self._entries.setdefault(sequence, ConsoleEntry(sequence, category))
        return RowAddress(lambda change: self.update(sequence, change))

    def update(self, sequence: int, change: RowChange) -> None:
        with self._lock:
            entry = self._entries[sequence]
            old = entry.state
            entry.update(change)
            text = Text(safe_text(f"#{sequence:02} {entry.category} {entry.state}"))
        self._write(text, change.state is not None and old != change.state)

    def _refresh(self, _text: Text, _changed: bool) -> None:
        self._live.refresh()

    def _transition(self, text: Text, changed: bool) -> None:
        if changed:
            self._console.print(text)

    def notify(self, sequence: int, state: str) -> None:
        with self._lock:
            exists = sequence in self._entries
        if exists:
            self.update(sequence, RowChange(state=state, finished=state not in {"STOPPING", "SAVING"}))

    def render(self) -> Group:
        with self._lock:
            entries = [entry for _, entry in sorted(self._entries.items()) if entry.run_id is not None or entry.finished]
            active = [entry for entry in entries if not entry.finished]
            visible = active or entries
            capacity = max(1, self._console.height - 3)
            hidden = max(0, len(visible) - capacity)
            completed = sum(entry.finished for entry in entries)
            errors = sum(entry.state in {"FAILED", "ERROR"} for entry in entries)
            heading = (f"ALERT finished {completed}/{self._total} · active {len(active)} · "
                       f"queued {self._total - len(entries)} · errors {errors} · "
                       f"vLLM {self._server().get('state', 'n/a')} · hidden {hidden}")
            return Group(Text(heading, no_wrap=True, overflow="ellipsis"), row_table(visible[:capacity], self._console.width))
