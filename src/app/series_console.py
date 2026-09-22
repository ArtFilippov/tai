"""One stderr Live display, updated by addressed journals rather than event polling."""

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from types import TracebackType
from typing import Self, TextIO, override

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from core.runs.console_ports import ConsoleRow, RowChange

from .console_format import ConsoleEntry, row_table, safe_text

DEFAULT_LINES_PER_ATTACK = 5


def validate_lines_per_attack(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("lines_per_attack must be a positive integer")


@dataclass
class _Rows:
    total: int
    lines_per_attack: int
    entries: dict[int, ConsoleEntry] = field(default_factory=dict[int, ConsoleEntry])
    slots: list[int | None] = field(default_factory=list[int | None])
    shown: set[int] = field(default_factory=set[int])

    def admitted(self) -> list[ConsoleEntry]:
        return [entry for entry in self.entries.values() if entry.run_id is not None or entry.finished]

    def visible(self, height: int) -> list[ConsoleEntry | None]:
        capacity = max(1, (height - 2) // self.lines_per_attack)
        self.slots = self.slots[:capacity] + [None] * max(0, capacity - len(self.slots))
        waiting = iter(self._waiting())
        for index, sequence in enumerate(self.slots):
            if sequence is None or self.entries[sequence].finished:
                if (entry := next(waiting, None)) is not None:
                    self.slots[index] = entry.sequence
                    self.shown.add(entry.sequence)
        return [self.entries[sequence] if sequence is not None else None for sequence in self.slots]

    def _waiting(self) -> list[ConsoleEntry]:
        return [entry for entry in self.admitted() if entry.sequence not in self.slots
                and (not entry.finished or entry.sequence not in self.shown)]

    def heading(self, visible: list[ConsoleEntry | None]) -> str:
        entries = self.admitted()
        active = sum(not entry.finished for entry in entries)
        hidden = active - sum(entry is not None and not entry.finished for entry in visible)
        errors = sum(entry.state in {"FAILED", "ERROR"} for entry in entries)
        heading = f"ALERT {len(entries) - active}/{self.total} · active {active} · queued {self.total - len(entries)}"
        if errors:
            heading += f" · errors {errors}"
        if hidden:
            heading += f" · hidden {hidden}"
        return heading


@dataclass(frozen=True)
class RowAddress(ConsoleRow):
    deliver: Callable[[RowChange], None]

    @override
    def update(self, change: RowChange) -> None:
        self.deliver(change)


class SeriesConsole:
    def __init__(
        self, total: int, *, backend: str = "unknown", lines_per_attack: int = DEFAULT_LINES_PER_ATTACK, stream: TextIO | None = None,
    ) -> None:
        super().__init__()
        validate_lines_per_attack(lines_per_attack)
        self._backend = backend
        self._rows = _Rows(total, lines_per_attack)
        self._console = Console(file=stream or sys.stderr, markup=False, highlight=False)
        self._lock = RLock()
        self._live = Live(console=self._console, get_renderable=self.render, refresh_per_second=10,
                          redirect_stdout=False, redirect_stderr=False, vertical_overflow="crop")
        self._write = self._refresh if self._console.is_terminal else self._transition

    def __enter__(self) -> Self:
        self._console.print(Text(safe_text(f"Backend: {self._backend}")))
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
            _ = self._rows.entries.setdefault(sequence, ConsoleEntry(sequence, category))
        return RowAddress(lambda change: self.update(sequence, change))

    def update(self, sequence: int, change: RowChange) -> None:
        with self._lock:
            entry = self._rows.entries[sequence]
            old = entry.display_state()
            entry.update(change)
            _ = self._rows.visible(self._console.height)
            text = Text(safe_text(f"#{sequence:02} {entry.category} {entry.display_state()}"))
            changed = change.state is not None and old != entry.display_state()
        self._write(text, changed)

    def _refresh(self, _text: Text, _changed: bool) -> None:
        self._live.refresh()

    def _transition(self, text: Text, changed: bool) -> None:
        if changed:
            self._console.print(text)

    def notify(self, sequence: int, state: str) -> None:
        with self._lock:
            exists = sequence in self._rows.entries
        if exists:
            self.update(sequence, RowChange(state=state, finished=state not in {"STOPPING", "SAVING"}))

    def render(self) -> Group:
        with self._lock:
            visible = self._rows.visible(self._console.height)
            return Group(
                Text(self._rows.heading(visible), no_wrap=True, overflow="ellipsis"),
                row_table(visible, self._console, self._rows.lines_per_attack),
            )
