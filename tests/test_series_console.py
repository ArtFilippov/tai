"""Terminal cells, plain-text content, independent rows, and redirected stderr."""

from io import StringIO

import pytest
from rich.cells import cell_len
from rich.console import Console

from app.console_format import ConsoleEntry, row_table
from app.series_console import SeriesConsole
from core.runs.console_ports import DisplayExchange, RowChange


@pytest.mark.parametrize("width", [12, 40, 80, 180])
def test_one_physical_row_per_run_at_terminal_width(width: int) -> None:
    entries = [ConsoleEntry(1, "category"), ConsoleEntry(2, "汉字 category")]
    for entry in entries:
        entry.update(RowChange(state="RUNNING", method="exchange[Target]", exchange=DisplayExchange(
            "Attacker", "a", "[bold] 汉字\n\x1b[31mREQUEST" * 10, "RESPONSE" * 10,
        )))
        entry.update(RowChange(exchange=DisplayExchange("Target", "t", "TARGET\rrequest", "ANSWER")))
    stream = StringIO()
    console = Console(file=stream, width=width, color_system=None, force_terminal=False)
    console.print(row_table(entries, width))
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2 and all(cell_len(line) <= width for line in lines)
    assert "\x1b" not in stream.getvalue()
    if width >= 80:
        assert all("A:" in line and "T:" in line and "RESP" in line and "ANSWER" in line for line in lines)


def test_non_tty_output_and_short_terminal_summary_have_no_control_codes() -> None:
    stream = StringIO()
    with SeriesConsole(32, stream=stream) as display:
        for sequence in range(1, 33):
            row = display.row(sequence, f"category-{sequence}")
            row.update(RowChange(state="RUNNING", run_id=str(sequence), method="assessment"))
        display.notify(1, "SAVING")
        display.notify(1, "COMPLETED")
    output = stream.getvalue()
    assert "finished 1/32" in output and "active 31" in output
    assert "hidden 9" in output
    assert "\x1b" not in output


def test_old_response_cannot_be_assigned_to_new_request() -> None:
    entry = ConsoleEntry(1, "category")
    entry.update(RowChange(exchange=DisplayExchange("Target", "first", "old", "old answer")))
    entry.update(RowChange(exchange=DisplayExchange("Target", "second", "new")))
    entry.update(RowChange(exchange=DisplayExchange("Target", "first", "old", "late old answer")))
    assert entry.exchanges["Target"].call_id == "second"
    assert entry.exchanges["Target"].response is None
