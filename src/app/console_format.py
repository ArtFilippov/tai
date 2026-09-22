"""Cell-width-aware plain-text rendering of directly supplied diagnostic rows."""

import re
import unicodedata
from dataclasses import dataclass, field

from rich.table import Table
from rich.text import Text

from core.runs.console_ports import DisplayExchange, RowChange

_ESCAPES = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]")
COMPACT_WIDTH = 50


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

    def pair(self, role: str, width: int) -> str:
        if (exchange := self.exchanges.get(role)) is None:
            return "—"
        request = Text(safe_text(exchange.request))
        response = Text(safe_text(exchange.response) if exchange.response is not None else "… waiting")
        request.truncate(max(1, (width - 3) // 2), overflow="ellipsis")
        response.truncate(max(1, (width - 3) // 2), overflow="ellipsis")
        return request.plain + " → " + response.plain


def row_table(entries: list[ConsoleEntry], width: int) -> Table:
    table = Table.grid(expand=True, padding=(0, 1))
    if width < COMPACT_WIDTH:
        table.add_column(no_wrap=True, overflow="ellipsis", width=max(1, width))
        for entry in entries:
            table.add_row(Text(safe_text(f"#{entry.sequence:02} {entry.state} {entry.method}")))
        return table
    for ratio in (2, 3, 4, 4):
        table.add_column(ratio=ratio, no_wrap=True, overflow="ellipsis", min_width=1)
    for entry in entries:
        pair_width = max(1, (width - 6) * 4 // 13 - 3)
        values = (f"#{entry.sequence:02} {entry.category}", f"{entry.state} {entry.method}",
                  "A: " + entry.pair("Attacker", pair_width), "T: " + entry.pair("Target", pair_width))
        table.add_row(*(Text(safe_text(value), no_wrap=True, overflow="ellipsis") for value in values))
    table.width = max(width, 1)
    return table
