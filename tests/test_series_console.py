"""Fixed attack slots, wrapped role columns, plain text, and startup metadata."""

from io import StringIO
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import override

import pytest
from rich.cells import cell_len
from rich.console import Console

from app import series
from app.console_format import ConsoleEntry, row_table
from app.configuration import load_profile
from app.series_console import SeriesConsole
from core.chats.openai import OpenAIChat
from core.runs.console_ports import DisplayExchange, RowChange


@pytest.mark.parametrize("width", [1, 8, 9, 12, 40, 80, 180])
@pytest.mark.parametrize("height", [1, 3, 5])
def test_fixed_height_blocks_at_terminal_width(width: int, height: int) -> None:
    entries = [ConsoleEntry(1, "category"), ConsoleEntry(2, "汉字 category")]
    for entry in entries:
        entry.update(RowChange(state="RUNNING", method="exchange[Target]", exchange=DisplayExchange(
            "Attacker", "a", "BOILERPLATE", "NOT_SENT",
        )))
        entry.update(RowChange(exchange=DisplayExchange("Target", "t", "TARGET [bold] 汉字\n\x1b[31mREQUEST " * 30, "ANSWER")))
    stream = StringIO()
    console = Console(file=stream, width=width, color_system=None, force_terminal=False)
    console.print(row_table([*entries], console, height))
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1 + 2 * height and all(cell_len(line) == width for line in lines)
    assert "\x1b" not in stream.getvalue()
    assert "BOILERPLATE" not in stream.getvalue() and "NOT_SENT" not in stream.getvalue()
    if width >= 80:
        assert "Attacker" in lines[0] and "Target" in lines[0]
        assert all("TARGET" in lines[index] and "ANSWER" in lines[index] for index in (1, 1 + height))
        assert "…" in lines[height] and "…" in lines[-1]


def test_non_tty_output_and_short_terminal_summary_have_no_control_codes() -> None:
    stream = StringIO()
    display = SeriesConsole(32, stream=stream, backend="Transformers", lines_per_attack=3)
    display._console.size = (120, 12)
    with display:
        for sequence in range(1, 33):
            row = display.row(sequence, f"category-{sequence}")
            row.update(RowChange(state="RUNNING", run_id=str(sequence), method="assessment"))
        display.notify(1, "SAVING")
        display.notify(1, "SUCCESS")
    output = stream.getvalue()
    assert "ALERT 1/32" in output and "active 31" in output
    assert "hidden 28" in output and "УСПЕХ" in output
    assert output.count("Backend: Transformers") == 1 and "vLLM" not in output
    assert "\x1b" not in output


def test_old_response_cannot_be_assigned_to_new_request() -> None:
    entry = ConsoleEntry(1, "category")
    entry.update(RowChange(exchange=DisplayExchange("Target", "first", "old", "old answer")))
    entry.update(RowChange(exchange=DisplayExchange("Target", "second", "new")))
    entry.update(RowChange(exchange=DisplayExchange("Target", "first", "old", "late old answer")))
    assert entry.exchanges["Target"].call_id == "second"
    assert entry.exchanges["Target"].response is None


def _frame(display: SeriesConsole) -> list[str]:
    with display._console.capture() as captured:
        display._console.print(display.render())
    return captured.get().splitlines()


def test_updates_completion_and_reuse_preserve_other_slot_positions() -> None:
    display = SeriesConsole(4, stream=StringIO(), lines_per_attack=4)
    display._console.size = (100, 14)
    for sequence in range(1, 4):
        display.row(sequence, "category").update(RowChange(state="RUNNING", run_id=str(sequence)))
    before = _frame(display)
    display.update(2, RowChange(method="exchange[Target]", exchange=DisplayExchange("Target", "call", "WORD " * 100)))
    pending = _frame(display)
    display.update(2, RowChange(exchange=DisplayExchange("Target", "call", "WORD " * 100, "ANSWER " * 100)))
    answered = _frame(display)
    display.notify(1, "SAVING")
    display.notify(1, "SUCCESS")
    completed = _frame(display)
    display.row(4, "category").update(RowChange(state="RUNNING", run_id="4"))
    reused = _frame(display)
    assert all(len(frame) == 14 for frame in (before, pending, answered, completed, reused))
    assert before[10:] == pending[10:] == answered[10:] == completed[10:] == reused[10:]
    assert answered[6:10] == completed[6:10] == reused[6:10]
    assert "УСПЕХ" in completed[2] and reused[2].startswith("04 ")
    assert reused[6].startswith("02 ") and reused[10].startswith("03 ")


def test_hidden_attack_takes_freed_slot_and_resize_preserves_visible_positions() -> None:
    display = SeriesConsole(4, stream=StringIO(), lines_per_attack=3)
    display._console.size = (100, 8)
    for sequence in range(1, 4):
        display.row(sequence, "category").update(RowChange(state="RUNNING", run_id=str(sequence)))
    before = _frame(display)
    assert "hidden 1" in before[0]
    display.notify(1, "STOPPED")
    promoted = _frame(display)
    assert promoted[2].startswith("03 ") and promoted[5:] == before[5:]
    display._console.size = (100, 11)
    expanded = _frame(display)
    assert expanded[2:8] == promoted[2:8] and all(not line.strip() for line in expanded[8:])


def test_target_pair_survives_attacker_and_evaluator_calls() -> None:
    display = SeriesConsole(1, stream=StringIO(), lines_per_attack=3)
    display._console.size = (120, 5)
    row = display.row(1, "category")
    row.update(RowChange(state="RUNNING", run_id="run", exchange=DisplayExchange("Target", "t", "SENT", "RECEIVED")))
    for role in ("Attacker", "Evaluator"):
        row.update(RowChange(method=f"exchange[{role}]", exchange=DisplayExchange(role, "other", "INSTRUCTIONS", "yes")))
        frame = "\n".join(_frame(display))
        assert f"exchange[{role}]" in frame and "SENT" in frame and "RECEIVED" in frame
        assert "INSTRUCTIONS" not in frame and "yes" not in frame


def test_empty_cells_and_short_answers_leave_reserved_lines_blank() -> None:
    stream = StringIO()
    console = Console(file=stream, width=100)
    entry = ConsoleEntry(1, "category", state="RUNNING")
    entry.update(RowChange(exchange=DisplayExchange("Target", "call", "short", "answer")))
    console.print(row_table([entry, None], console, 3))
    lines = stream.getvalue().splitlines()
    assert len(lines) == 7 and "short" in lines[1] and "answer" in lines[1]
    assert all(not line.strip() for line in lines[2:])


@pytest.mark.parametrize(("state", "label"), [
    ("COMPLETED", "SAVING"), ("SUCCESS", "УСПЕХ"), ("UNSUCCESSFUL", "НЕУСПЕХ"),
    ("UNEVALUATED", "НЕТ ОЦЕНКИ"), ("STOPPED", "STOPPED"), ("ERROR", "ERROR"),
])
def test_display_labels(state: str, label: str) -> None:
    display = SeriesConsole(1, stream=StringIO(), lines_per_attack=2)
    display._console.size = (100, 4)
    display.row(1, "category").update(RowChange(state=state, run_id="run"))
    assert label in "\n".join(_frame(display))


def test_failure_before_bind_is_visible_without_recycling_finished_entries() -> None:
    display = SeriesConsole(2, stream=StringIO(), lines_per_attack=2)
    display._console.size = (100, 4)
    _ = display.row(1, "category")
    display.notify(1, "ERROR")
    assert "ERROR" in "\n".join(_frame(display))
    display.row(2, "category").update(RowChange(state="RUNNING", run_id="run"))
    display.notify(2, "SUCCESS")
    assert _frame(display)[2].startswith("02 ")
    assert _frame(display)[2].startswith("02 ")


class _TerminalStream(StringIO):
    @override
    def isatty(self) -> bool:
        return True


def test_live_refresh_keeps_backend_outside_redrawn_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM", "xterm")
    stream = _TerminalStream()
    display = SeriesConsole(1, backend="Transformers", lines_per_attack=3, stream=stream)
    display._console.size = (100, 8)
    with display:
        row = display.row(1, "category")
        row.update(RowChange(state="RUNNING", run_id="run", exchange=DisplayExchange("Target", "call", "Question")))
        row.update(RowChange(exchange=DisplayExchange("Target", "call", "Question", "Answer")))
        display.notify(1, "SUCCESS")
    output = stream.getvalue()
    assert output.count("Backend: Transformers") == 1 and "\x1b" in output
    assert "Question" in output and "Answer" in output and "УСПЕХ" in output
    assert "Backend" not in "\n".join(_frame(display))


@pytest.mark.parametrize(("filename", "label"), [
    ("fitd-alert-transformers.toml", "Transformers"), ("fitd-alert-vllm.toml", "vLLM"),
])
def test_backend_label_uses_configured_roles(filename: str, label: str) -> None:
    with closing(load_profile(Path("examples") / filename)) as profile:
        assert series._backend(profile) == label
        chat = OpenAIChat(base_url="http://localhost:1/v1", model="unused")
        mixed = replace(profile, attacker=chat)
        assert series._backend(mixed) == f"Target: {label} · Attacker: OpenAI-compatible · Evaluator: {label}"
