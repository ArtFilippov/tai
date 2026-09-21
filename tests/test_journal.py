"""Independent observation of real FITD, including poisoned storage boundaries."""

import json
from collections import Counter
from pathlib import Path
from threading import Event
from typing import Unpack

import pytest
from journal_assertions import at, mapping, released_worker, result_of, run_directory
from journal_failures import broken_journal, failing_journal_factory
from journal_scenarios import (
    assert_branch_links, assert_evaluation_links, assert_initialization,
    count_native_checks, keyed_events, participants,
)

import attacks.fitd.execution as implementation

from tai import (
    EvaluatorSetup,
    ContainsTextEvaluator,
    FITDExperiment,
    FITDUser,
    LLMChat,
    Message,
    RunJournal,
    UserState,
    capture_run,
    save_run,
)
from app.configuration import RoleProfile
from app.preflight import run_preflight
from app.terminal import run_foreground
from core.evaluation.model_evaluator import MODEL_RUBRIC_VERSION
from core.runs.journal_lifecycle import JournalOptions
from attacks.fitd.native import GarakException


@pytest.mark.parametrize(
    ("targets", "continuations"),
    [
        (["One", "Two", "Three", "Four", "CONTROL"], ["yes"] * 5),
        (["Opening", "I cannot help with that"], ["yes", "Rephrase"]),
        (["I cannot help with that"], ["Safer"]),
        (["Opening", "Off topic", "Correction"], ["yes", "no", "no"]),
    ],
)
def test_wrapping_preserves_native_checks_calls_order_and_presented_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    targets: list[str], continuations: list[str],
):
    counts = count_native_checks(monkeypatch)
    snapshots: list[object] = []
    for user_type in (FITDUser, FITDExperiment):
        counts.clear()
        target, attacker, base, evaluator, trace = participants(targets, continuations)
        user = (
            FITDExperiment(target, attacker, base, evaluator, journal_dir=tmp_path,
                           max_calls_per_conv=len(targets))
            if user_type is FITDExperiment else
            FITDUser(target, attacker, base, evaluator, max_calls_per_conv=len(targets))
        )
        user.start()
        assert user.wait(10)
        status = user.status()
        assert status.state is UserState.COMPLETED, status.reason
        snapshots.append(
            (
                trace,
                counts.copy(),
                status.reason,
                status.progress,
                result_of(user).to_string() if user.result() else None,
                status.evaluation.success if status.evaluation else None,
                tuple(t.message.to_string() for t in target.history()),
            )
        )
        if isinstance(user, FITDExperiment):
            events = user.journal.events()
            assert len([e for e in events if e.kind == "call_started"]) == len(trace)
            checks = [mapping(e.data) for e in events if e.kind == "local_decision"]
            assert counts["_is_refusal"] == sum(
                c["decision"] == "refusal" for c in checks
            )
            assert counts["_is_response_aligned"] == sum(
                c["decision"] == "alignment" for c in checks
            )
        else:
            assert not hasattr(user, "events")
            assert not (run_directory(user, tmp_path) / "events.jsonl").exists()
    assert snapshots[0] == snapshots[1]


def test_real_fitd_journal_connects_initialization_rollback_and_model_verdicts(
    tmp_path: Path,
):
    target, attacker, base, evaluator, trace = participants(
        [
            "One",
            "Two",
            "Three",
            "Four",
            "Negative",
            "I cannot help with that",
            "CONTROL",
        ],
        ["yes"] * 5 + ["Rephrase", "yes"],
        ['{"success": false}', '{"success": true}'],
    )
    experiment = FITDExperiment(target, attacker, base, evaluator, journal_dir=tmp_path)
    experiment.start()
    journal = experiment.journal
    assert journal.wait(10)  # No User.wait required to finalize the journal.
    assert journal.status().reason == "evaluator_success"
    events = journal.events()
    assert_initialization(events, journal.status().run_id)
    starts = keyed_events(events, "call_started", "call_id")
    ends = keyed_events(events, "call_completed", "call_id")
    assert len(starts) == len(ends) == len(trace)
    assert_branch_links(events, ends)
    assert_evaluation_links(events, starts, result_of(experiment))
    copied = mapping(events[-1].data)
    copied["state"] = "corrupted"
    assert at(events[-1].data, "state") == "COMPLETED"
    assert experiment.wait(10)


@pytest.mark.parametrize("error_type", [OSError, GarakException])
@pytest.mark.parametrize(
    ("kind", "role", "expected"),
    [
        ("run_started", None, (0, 0, 0)),
        ("checkpoint_created", None, (0, 0, 0)),
        ("call_started", "Attacker", (0, 0, 0)),
        ("call_completed", "Attacker", (0, 1, 0)),
        ("local_decision", "Attacker", (0, 1, 0)),
        ("call_started", "Target", (0, 4, 0)),
        ("call_completed", "Target", (1, 4, 0)),
        ("branch_selected", None, (1, 5, 0)),
        ("result_published", None, (1, 5, 0)),
        ("evaluation_started", None, (5, 9, 0)),
        ("call_started", "Evaluator", (5, 9, 0)),
        ("call_completed", "Evaluator", (5, 9, 1)),
        ("evaluation_completed", None, (5, 9, 1)),
        ("run_finished", None, (5, 9, 1)),
    ],
)
def test_storage_failure_prevents_next_work_and_survives_library_handlers_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, role: str | None,
    expected: tuple[int, int, int], error_type: type[Exception],
):
    failure = error_type("journal storage failed")
    journal_factory, attempted, streams = broken_journal(monkeypatch, kind, role, failure)
    target, attacker, base, evaluator, trace = participants(
        ["One", "Two", "Three", "Four", "CONTROL"], ["yes"] * 5
    )
    experiment = FITDExperiment(
        target,
        attacker,
        base,
        evaluator,
        journal_dir=tmp_path,
        journal_factory=journal_factory,
    )
    experiment.start()
    assert experiment.wait(10)
    journal = experiment.journal
    assert journal.finished and journal.error is failure
    assert journal.status().state is experiment.status().state is UserState.FAILED
    assert (
        experiment.status().reason == f"{error_type.__name__}: journal storage failed"
    )
    assert attempted[-1].kind == kind
    assert all(stream.source.closed for stream in streams)
    assert tuple(Counter(role for role, _ in trace)[r]
                 for r in ("Target", "Attacker", "Evaluator")) == expected
    assert not journal.events(after=attempted[-1].sequence)
    with pytest.raises(error_type) as raised:
        journal.record("retry")
    assert raised.value is failure
    assert evaluator.chat is not None
    assert all(chat._execution.owner is None for chat in (target, attacker, evaluator.chat))
    assert_saved_failure(experiment, tmp_path, expected[0])
    if kind == "call_completed" and role == "Target":
        assert target.history()[-1].message.to_string() == "One"
        assert experiment.result() is None
    if kind == "result_published":
        assert experiment.result() is target.history()[-1].message
        assert (
            run_directory(experiment, tmp_path) / "result.txt"
        ).read_text() == "One"


def assert_saved_failure(experiment: FITDExperiment, root: Path, target_calls: int) -> None:
    directory = save_run(capture_run(experiment, journal=experiment.journal), root)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["state"] == "FAILED"
    assert metadata["reason"] == experiment.status().reason
    assert metadata["progress"]["target_calls"] == target_calls


def assert_cleanup_pending(experiment: FITDExperiment, journal: RunJournal) -> None:
    assert experiment.status().state is UserState.COMPLETED
    assert not journal.finished and not journal.wait(0)
    assert not any(event.kind == "run_finished" for event in journal.events())
    with pytest.raises(RuntimeError, match="active run"):
        experiment.start()


def test_completion_follows_resource_release_and_retained_journal_is_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    entered, release = Event(), Event()
    release_chats = implementation.release_chats

    def hold_release(chats: tuple[LLMChat, ...], owner: object) -> None:
        entered.set()
        assert release.wait(5)
        release_chats(chats, owner)

    monkeypatch.setattr(implementation, "release_chats", hold_release)
    targets = iter(["Same", "Same"])
    attackers = iter(["Initial", "L1", "L2", "L3", "yes"] * 2)

    experiment = FITDExperiment(
        LLMChat(lambda turns: next(targets)),
        LLMChat(lambda turns: next(attackers)),
        Message("Puzzle"),
        ContainsTextEvaluator("CONTROL"),
        journal_dir=tmp_path,
        max_calls_per_conv=1,
    )
    experiment.start()
    first = experiment.journal
    with released_worker(experiment, release):
        assert entered.wait(5)
        assert_cleanup_pending(experiment, first)
    assert first.wait(5)
    assert experiment.wait(5)
    records = first.events()
    directory = run_directory(experiment, tmp_path)
    old_bytes = (directory / "events.jsonl").read_bytes()
    experiment.start()
    second = experiment.journal
    assert second is not first
    assert second.wait(5)
    assert experiment.wait(5)
    assert experiment.journals == (first, second)
    assert first.events() == records
    assert (directory / "events.jsonl").read_bytes() == old_bytes
    assert first.status().run_id != second.status().run_id
    assert first.status().result_id != second.status().result_id


def test_cli_and_preflight_fail_on_journal_errors_without_extra_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
):
    create_journal = failing_journal_factory(
        monkeypatch, OSError("cannot write journal"), lambda event: event.kind == "call_started",
    )
    target, attacker, base, evaluator, trace = participants(["Unused"], ["yes"])
    experiment = FITDExperiment(
        target,
        attacker,
        base,
        evaluator,
        journal_dir=tmp_path,
        journal_factory=create_journal,
    )
    assert run_foreground(experiment, artifacts_dir=tmp_path) == 1
    assert "cannot write journal" in capsys.readouterr().out
    assert not trace
    journals: list[RunJournal] = []

    def factory(**kwargs: Unpack[JournalOptions]) -> RunJournal:
        journal = create_journal(**kwargs)
        journals.append(journal)
        return journal

    profile = RoleProfile(target, attacker, evaluator, 1)
    with pytest.raises(OSError, match="cannot write journal"):
        run_preflight(profile, journal_factory=factory)
    assert not trace
    assert journals[0].finished
    assert journals[0].status().state is UserState.FAILED
    assert journals[0].error is not None


def test_cleanup_failure_is_the_final_journal_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    release_chats = implementation.release_chats

    def fail_cleanup(chats: tuple[LLMChat, ...], owner: object) -> None:
        release_chats(chats, owner)
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(implementation, "release_chats", fail_cleanup)
    target, attacker, base, evaluator, _ = participants(["Opening"], ["yes"])
    experiment = FITDExperiment(
        target, attacker, base, evaluator, journal_dir=tmp_path, max_calls_per_conv=1
    )
    experiment.start()
    assert experiment.wait(10)
    journal = experiment.journal
    assert journal.finished and journal.error is None
    assert journal.status() == experiment.status()
    assert journal.status().state is UserState.FAILED
    assert at(journal.events()[-1].data, "reason") == "RuntimeError: cleanup failed"
    save_run(capture_run(experiment, journal=journal), tmp_path)
    metadata = json.loads(
        (run_directory(experiment, tmp_path) / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["state"] == "FAILED"
    assert metadata["reason"] == journal.status().reason


def test_preflight_reports_real_exchange_and_verdict_events(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    target = LLMChat(lambda turns: "READY")
    attacker = LLMChat(lambda turns: "READY")
    judge = LLMChat(lambda turns: '{"success": false}')
    profile = RoleProfile(
        target, attacker, EvaluatorSetup.model(judge, MODEL_RUBRIC_VERSION), 1
    )
    assert not run_preflight(profile, tmp_path / "report.json")
    data = json.loads(capsys.readouterr().out)
    assert data == json.loads((tmp_path / "report.json").read_text())
    events = data["events"]
    assert len([e for e in events if e["kind"] == "call_completed"]) == 3
    assert (
        next(
            e["data"]["success"] for e in events if e["kind"] == "evaluation_completed"
        )
        is False
    )
    assert events[-1]["kind"] == "run_finished"
