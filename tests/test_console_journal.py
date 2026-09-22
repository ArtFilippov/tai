"""Direct operation observation, nested stacks, redaction and fail-fast output."""

from functools import partial
from pathlib import Path
from threading import Event
from typing import override

import pytest
from fitd_support import releasing
from journal_scenarios import participants

from app.console_format import ConsoleEntry
from core.chats import ChatTurn, LLMChat
from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.messages import Message
from core.runs.console_journal import ConsoleRunJournal
from core.runs.console_ports import ConsoleRow, RowChange
from core.runs.journal import RunJournal
from core.runs.operations import Cause, Context, Exchange, WorkerOutcome
from core.runs.status import UserState, UserStatus
from tai import FITDExperiment


class RecordedRow(ConsoleRow):
    def __init__(self) -> None:
        super().__init__()
        self.entry = ConsoleEntry(1, "category")
        self.changes: list[RowChange] = []
        self.failure: Exception | None = None

    @override
    def update(self, change: RowChange) -> None:
        self.entry.update(change)
        self.changes.append(change)
        if self.failure is not None:
            raise self.failure


def test_direct_nested_method_and_correlated_role_pairs(tmp_path: Path) -> None:
    row = RecordedRow()
    journal = ConsoleRunJournal(row, redactor=DiagnosticRedactor(["secret"]))
    journal.bind("run")
    journal.begin("run", tmp_path / "run")
    context = Context("run", "Attacker", "chat", Cause("test"))
    request, response = Message("secret question"), Message("secret answer")

    def exchange() -> Exchange:
        assert row.entry.method == "exchange[Attacker]"
        assert "Attacker" not in row.entry.exchanges
        return Exchange(context, "first", request, response)

    def nested() -> Exchange:
        result = journal.call("exchange", exchange, context=context, call_id="first", message=request)
        assert row.entry.method == "FITDProbe.check_alignment"
        return result

    result = journal.call("FITDProbe.check_alignment", nested)
    assert result.response is response and result.request is request
    assert not row.entry.method
    assert "secret" not in str(row.changes) and response.text == "secret answer"
    target = Context("run", "Target", "target", Cause("test"))
    journal.call("exchange", lambda: Exchange(target, "target", request, response), context=target, call_id="target", message=request)

    def pending() -> Exchange:
        assert row.entry.exchanges["Attacker"].call_id == "first"
        assert row.entry.exchanges["Attacker"].response is not None
        assert row.entry.exchanges["Target"].response is not None
        return Exchange(context, "second", Message("new request"), Message("new response"))

    journal.call("exchange", pending, context=context, call_id="second", message=Message("new request"))
    assert row.entry.exchanges["Attacker"].call_id == "second"
    journal.finish(WorkerOutcome(UserStatus(state=UserState.COMPLETED, run_id="run"), None, None, None))
    assert row.entry.state == "COMPLETED" and not row.entry.method
    assert (tmp_path / "run" / "events.jsonl").exists()


@pytest.mark.parametrize("fails", [False, True])
def test_completed_target_pair_survives_pending_or_failed_exchange(tmp_path: Path, fails: bool) -> None:
    row = RecordedRow()
    journal = ConsoleRunJournal(row)
    journal.bind("run")
    journal.begin("run", tmp_path / "run")
    context = Context("run", "Target", "chat", Cause("test"))
    first = Exchange(context, "first", Message("first request"), Message("first response"))
    second = Exchange(context, "second", Message("second request"), Message("second response"))
    journal.call("exchange", lambda: first, context=context, call_id=first.call_id, message=first.request)

    def exchange() -> Exchange:
        assert row.entry.method == "exchange[Target]"
        assert row.entry.cells()[3:] == (first.request.text, first.response.text)
        if fails:
            raise OSError("exchange failed")
        return second

    if fails:
        with pytest.raises(OSError, match="exchange failed"):
            journal.call("exchange", exchange, context=context, call_id=second.call_id, message=second.request)
    else:
        journal.call("exchange", exchange, context=context, call_id=second.call_id, message=second.request)
    expected = first if fails else second
    assert row.entry.cells()[3:] == (expected.request.text, expected.response.text)
    assert row.entry.exchanges["Target"].call_id == expected.call_id and not row.entry.method
    assert all(change.exchange is None or change.exchange.response is not None for change in row.changes)
    state = UserState.FAILED if fails else UserState.COMPLETED
    journal.finish(WorkerOutcome(UserStatus(state=state, run_id="run"), None, None, None))
    assert row.entry.cells()[3:] == (expected.request.text, expected.response.text)


def test_primary_exception_survives_failed_unwind_and_resources_release(tmp_path: Path) -> None:
    row = RecordedRow()
    journal = ConsoleRunJournal(row)
    journal.bind("run")
    journal.begin("run", tmp_path / "run")
    original = ValueError("operation failed")
    render_error = OSError("output failed")

    def invoke() -> None:
        row.failure = render_error
        raise original

    with pytest.raises(ValueError) as caught:
        journal.call("check_alignment", invoke)
    assert caught.value is original and journal.error is render_error
    with pytest.raises(OSError):
        journal.call("next", lambda: pytest.fail("poisoned observer admitted work"))
    with pytest.raises(OSError):
        journal.finish(WorkerOutcome(UserStatus(state=UserState.FAILED, run_id="run"), None, original, None))
    assert journal.finished and not row.entry.method


def test_real_fitd_pair_visible_only_after_response_and_matches_unobserved_result(tmp_path: Path) -> None:
    row = RecordedRow()
    entered, release = Event(), Event()
    target, attacker, base, setup, _ = participants(["Accepted"], ["yes"])

    def generate(_turns: tuple[ChatTurn, ...]) -> str:
        entered.set()
        assert release.wait(5)
        return "Accepted"

    observed = FITDExperiment(LLMChat(generate), attacker, base, setup, max_calls_per_conv=1,
                              journal_dir=tmp_path, journal_factory=partial(ConsoleRunJournal, row))
    assert not row.changes
    observed.start()
    with releasing(observed, release):
        assert entered.wait(5)
        assert row.entry.method == "exchange[Target]"
        assert row.entry.run_id == observed.status().run_id
        assert row.entry.cells()[3:] == ("", "") and "Target" not in row.entry.exchanges
        assert row.entry.exchanges["Attacker"].response is not None
    assert not row.entry.method and row.entry.state == "COMPLETED"
    assert row.entry.exchanges["Target"].response == "Accepted"
    _, plain_attacker, plain_base, plain_setup, _ = participants(["Accepted"], ["yes"])
    plain = FITDExperiment(target, plain_attacker, plain_base, plain_setup, max_calls_per_conv=1, journal_factory=RunJournal)
    plain.start()
    assert plain.wait(5)
    plain_result, observed_result = plain.result(), observed.result()
    assert plain_result is not None and observed_result is not None
    assert plain.status().progress == observed.status().progress
    assert plain_result.text == observed_result.text
