"""Cell-width-aware plain-text rendering of directly supplied diagnostic rows."""

import re
import unicodedata
from dataclasses import dataclass, field

from rich.console import Console
from rich.text import Text

from core.runs.console_ports import DisplayExchange, RowChange

_ESCAPES = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]")
_HEADERS = ("#", "Состояние", "Текущая функция", "Attacker", "Target")
_MIN_TABLE_WIDTH = 2 * len(_HEADERS) - 1
_STATES = {"COMPLETED": "SAVING", "SUCCESS": "УСПЕХ", "UNSUCCESSFUL": "НЕУСПЕХ", "UNEVALUATED": "НЕТ ОЦЕНКИ"}


def safe_text(value: str) -> str:
    plain = " ".join(_ESCAPES.sub("", value).split())
    return "".join(character for character in plain if not unicodedata.category(character).startswith("C"))


@dataclass
class ConsoleEntry:
    sequence: int
    category: str
    state: str = "IDLE"
    method: str = ""
    run_id: str | None = None
    exchanges: dict[str, DisplayExchange] = field(default_factory=dict[str, DisplayExchange])
    finished: bool = False

    def update(self, change: RowChange) -> None:
        for name in ("state", "method", "run_id", "finished"):
            if (value := getattr(change, name)) is not None:
                setattr(self, name, value)
        if (exchange := change.exchange) is not None and exchange.role in {"Attacker", "Target"}:
            previous = self.exchanges.get(exchange.role)
            if exchange.response is None or previous is None or previous.call_id == exchange.call_id:
                self.exchanges[exchange.role] = exchange

    def cells(self) -> tuple[str, ...]:
        exchange = self.exchanges.get("Target")
        request = exchange.request if exchange is not None else ""
        response = (exchange.response if exchange.response is not None else "…") if exchange is not None else ""
        return (f"{self.sequence:02}", self.display_state(), self.method, request, response)

    def display_state(self) -> str:
        return _STATES.get(self.state, self.state)


def _column_widths(width: int) -> tuple[int, ...]:
    if width < _MIN_TABLE_WIDTH:
        return (max(1, width),)
    available = width - 4
    metadata = (max(1, min(3, available // 16)), max(1, min(10, available // 7)), max(1, min(24, available // 5)))
    remaining = available - sum(metadata)
    return (*metadata, remaining // 2, remaining - remaining // 2)


def _cell_lines(value: str, console: Console, width: int, height: int) -> list[str]:
    wrapped = list(Text(safe_text(value)).wrap(console, width, overflow="fold"))
    lines = wrapped[:height]
    if len(wrapped) > height:
        lines[-1].truncate(width - 1, overflow="crop")
        _ = lines[-1].append("…")
    lines.extend(Text() for _ in range(height - len(lines)))
    for line in lines:
        line.truncate(width, overflow="crop", pad=True)
    return [line.plain for line in lines]


def _block(values: tuple[str, ...], console: Console, height: int) -> list[str]:
    widths = _column_widths(console.width)
    if len(widths) == 1:
        values = (" ".join(values),)
    columns = [_cell_lines(value, console, width, height) for value, width in zip(values, widths, strict=True)]
    return [" ".join(row) for row in zip(*columns, strict=True)]


def row_table(entries: list[ConsoleEntry | None], console: Console, lines_per_attack: int) -> Text:
    lines = _block(_HEADERS, console, 1)
    for entry in entries:
        values = entry.cells() if entry is not None else ("",) * len(_HEADERS)
        lines.extend(_block(values, console, lines_per_attack))
    return Text("\n".join(lines), no_wrap=True, overflow="crop")
