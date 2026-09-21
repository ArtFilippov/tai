"""Independent observation of real FITD, including poisoned storage boundaries."""

import json
from collections import Counter
from pathlib import Path
from threading import Event

import pytest
from garak.exception import GarakException
from garak.probes.fitd import FITD

from tai import (
    FITDExperiment,
    FITDUser,
    LLMChat,
    Message,
    ModelEvaluator,
    RunJournal,
    UserState,
)
from tai._config import ModelEvaluatorProfile, RoleProfile
from tai._preflight import run_preflight
from tai.evaluators import MODEL_RUBRIC_VERSION
from tai.terminal import run_foreground


def participants(targets, continuations, verdicts=(' {"success": true}',)):
    trace = []

    def chat(role, replies):
        replies = iter(replies)

        def generate(turns):
            trace.append((role, tuple((t.role, t.message.to_string()) for t in turns)))
            return next(replies)

        return LLMChat(generate, system=role + " instructions")

    target = chat("Target", targets)
    attacker = chat("Attacker", ["Initial", "L1", "L2", "L3", *continuations])
    judge = chat("Evaluator", verdicts)
    base = Message("Solve the fictional puzzle")
    return target, attacker, base, ModelEvaluator(judge, base), trace


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
    tmp_path, monkeypatch, targets, continuations
):
    counts = Counter()

    def count(name):
        operation = getattr(FITD, name)

        def invoke(*args, **kwargs):
            counts[name] += 1
            return operation(*args, **kwargs)

        monkeypatch.setattr(FITD, name, invoke)

    for name in (
        "_is_refusal",
        "_is_response_aligned",
        "_is_realign_prompt",
        "generate_next_turn_attempt",
    ):
        count(name)
    snapshots = []
    for user_type in (FITDUser, FITDExperiment):
        counts.clear()
        target, attacker, base, evaluator, trace = participants(targets, continuations)
        user = user_type(
            target,
            attacker,
            base,
            evaluator,
            artifacts_dir=tmp_path,
            max_calls_per_conv=len(targets),
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
                user.result().to_string() if user.result() else None,
                status.evaluation.success if status.evaluation else None,
                tuple(t.message.to_string() for t in target.history()),
            )
        )
        if isinstance(user, FITDExperiment):
            events = user.journal.events()
            assert len([e for e in events if e.kind == "call_started"]) == len(trace)
            checks = [e.data for e in events if e.kind == "local_decision"]
            assert counts["_is_refusal"] == sum(
                c["decision"] == "refusal" for c in checks
            )
            assert counts["_is_response_aligned"] == sum(
                c["decision"] == "alignment" for c in checks
            )
        else:
            assert not hasattr(user, "events")
            assert not (tmp_path / status.run_id / "events.jsonl").exists()
    assert snapshots[0] == snapshots[1]


def test_real_fitd_journal_connects_initialization_rollback_and_model_verdicts(
    tmp_path,
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
    experiment = FITDExperiment(
        target, attacker, base, evaluator, artifacts_dir=tmp_path
    )
    experiment.start()
    journal = experiment.journal
    assert journal.wait(10)  # No User.wait required to finalize the journal.
    assert journal.status().reason == "evaluator_success"
    events = journal.events()
    assert events[0].kind == "run_started"
    assert events[-2].kind == "resources_released"
    assert events[-1].kind == "run_finished"
    assert [e.sequence for e in events] == list(range(1, len(events) + 1))
    assert all(e.run_id == journal.status().run_id for e in events)
    resets = [
        e.data
        for e in events
        if e.kind == "history_reset"
        and e.data["context"]["cause"]["reason"] == "run_started"
    ]
    assert [e["role"] for e in resets] == ["Target", "Attacker", "Evaluator"]
    initial = next(e for e in events if e.kind == "checkpoint_created")
    assert initial.data["context"]["cause"]["reason"] == "initial_checkpoint"
    assert initial.sequence < next(
        e.sequence for e in events if e.kind == "call_started"
    )
    starts = {e.data["call_id"]: e.data for e in events if e.kind == "call_started"}
    ends = {e.data["call_id"]: e.data for e in events if e.kind == "call_completed"}
    assert len(starts) == len(ends) == len(trace)
    exchanges = {
        e.data["checkpoint_id"]: e.data for e in events if e.kind == "target_exchange"
    }
    branches = [e.data for e in events if e.kind == "branch_selected"]
    assert [b["branch"] for b in branches] == ["continue"] * 5 + [
        "rephrase",
        "continue",
    ]
    for branch in branches:
        exchange = exchanges[branch["checkpoint_id"]]
        assert branch["response_id"] == ends[exchange["call_id"]]["response"]["id"]
        assert branch["attempt_id"] == exchange["attempt_id"]
        for check in branch["checks"]:
            assert check["context"] == branch["context"]
            if check["judgment"]:
                assert check["judgment"]["call_id"] in ends
    rollback = next(
        e.data
        for e in events
        if e.kind == "conversation_synchronized"
        and e.data["outcome"]["conversation"]["kind"] == "restored"
    )
    assert rollback["cause_id"] == branches[5]["branch_id"]
    assert rollback["from_checkpoint_id"] == branches[5]["checkpoint_id"]
    assert rollback["checkpoint_id"] == branches[4]["checkpoint_id"]
    evaluations = [e.data for e in events if e.kind == "evaluation_completed"]
    assert [e["success"] for e in evaluations] == [False, True]
    publications = [e.data for e in events if e.kind == "result_published"]
    assert publications[0]["change"] == "accepted"
    assert publications[-1]["change"] == "replaced"
    assert (
        publications[-1]["previous_evaluation"]["result_id"]
        == evaluations[0]["result_id"]
    )
    for evaluation in evaluations:
        publication = next(
            p for p in publications if p["result"]["id"] == evaluation["result_id"]
        )
        assert evaluation["checkpoint_id"] == publication["checkpoint_id"]
        assert evaluation["cause_id"] == publication["cause_id"]
        assert (
            sum(s.get("result_id") == evaluation["result_id"] for s in starts.values())
            == 1
        )
    assert experiment.result().id == evaluations[-1]["result_id"]
    copied = events[-1].data
    copied["state"] = "corrupted"
    assert events[-1].data["state"] == "COMPLETED"
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
    tmp_path, monkeypatch, kind, role, expected, error_type
):
    failure = error_type("journal storage failed")
    attempted = []
    streams = []
    open_file = Path.open

    class Stream:
        def __init__(self, source):
            self.source = source

        def write(self, text):
            return self.source.write(text)

        def flush(self):
            self.source.flush()

        def close(self):
            self.source.close()
            raise RuntimeError("secondary close error")

    def open_stream(path, *args, **kwargs):
        source = open_file(path, *args, **kwargs)
        if path.name == "events.jsonl":
            streams.append(Stream(source))
            return streams[-1]
        return source

    class BrokenJournal(RunJournal):
        def _write_event(self, event):
            attempted.append(event)
            if event.kind == kind and (role is None or event.data.get("role") == role):
                raise failure
            super()._write_event(event)

    monkeypatch.setattr(Path, "open", open_stream)
    target, attacker, base, evaluator, trace = participants(
        ["One", "Two", "Three", "Four", "CONTROL"], ["yes"] * 5
    )
    experiment = FITDExperiment(
        target,
        attacker,
        base,
        evaluator,
        artifacts_dir=tmp_path,
        journal_factory=BrokenJournal,
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
    actual = Counter(role for role, _ in trace)
    assert tuple(actual[r] for r in ("Target", "Attacker", "Evaluator")) == expected
    assert not journal.events(after=attempted[-1].sequence)
    with pytest.raises(error_type) as raised:
        journal.record("retry")
    assert raised.value is failure
    assert all(chat._owner is None for chat in (target, attacker, evaluator.chat))
    if kind == "call_completed" and role == "Target":
        assert target.history()[-1].message.to_string() == "One"
        assert experiment.result() is None
    if kind == "result_published":
        assert experiment.result() is target.history()[-1].message
        assert (
            tmp_path / experiment.status().run_id / "result.txt"
        ).read_text() == "One"


def test_completion_follows_resource_release_and_retained_journal_is_independent(
    tmp_path, monkeypatch
):
    import tai.fitd as implementation

    entered, release = Event(), Event()
    release_chats = implementation._release_chats

    def hold_release(chats, owner):
        entered.set()
        assert release.wait(5)
        release_chats(chats, owner)

    monkeypatch.setattr(implementation, "_release_chats", hold_release)
    targets = iter(["Same", "Same"])
    attackers = iter(["Initial", "L1", "L2", "L3", "yes"] * 2)
    from tai import ContainsTextEvaluator

    experiment = FITDExperiment(
        LLMChat(lambda turns: next(targets)),
        LLMChat(lambda turns: next(attackers)),
        Message("Puzzle"),
        ContainsTextEvaluator("CONTROL"),
        artifacts_dir=tmp_path,
        max_calls_per_conv=1,
    )
    experiment.start()
    first = experiment.journal
    try:
        assert entered.wait(5)
        assert experiment.status().state is UserState.COMPLETED
        assert not first.finished and not first.wait(0)
        assert not any(e.kind == "run_finished" for e in first.events())
        with pytest.raises(RuntimeError, match="active run"):
            experiment.start()
    finally:
        release.set()
    assert first.wait(5)
    assert experiment.wait(5)
    records = first.events()
    old_bytes = (tmp_path / first.status().run_id / "events.jsonl").read_bytes()
    experiment.start()
    second = experiment.journal
    assert second is not first
    assert second.wait(5)
    assert experiment.wait(5)
    assert experiment.journals == (first, second)
    assert first.events() == records
    assert (tmp_path / first.status().run_id / "events.jsonl").read_bytes() == old_bytes
    assert first.status().run_id != second.status().run_id
    assert first.status().result_id != second.status().result_id


def test_cli_and_preflight_fail_on_journal_errors_without_extra_calls(tmp_path, capsys):
    class BrokenJournal(RunJournal):
        def _write_event(self, event):
            if event.kind == "call_started":
                raise OSError("cannot write journal")
            super()._write_event(event)

    target, attacker, base, evaluator, trace = participants(["Unused"], ["yes"])
    experiment = FITDExperiment(
        target,
        attacker,
        base,
        evaluator,
        artifacts_dir=tmp_path,
        journal_factory=BrokenJournal,
    )
    assert run_foreground(experiment, artifacts_dir=tmp_path) == 1
    assert "cannot write journal" in capsys.readouterr().out
    assert trace == []
    journals = []

    def factory(**kwargs):
        journal = BrokenJournal(**kwargs)
        journals.append(journal)
        return journal

    profile = RoleProfile(
        target, attacker, ModelEvaluatorProfile(evaluator.chat, MODEL_RUBRIC_VERSION), 1
    )
    with pytest.raises(OSError, match="cannot write journal"):
        run_preflight(profile, journal_factory=factory)
    assert trace == []
    assert journals[0].finished
    assert journals[0].status().state is UserState.FAILED
    assert journals[0].error is not None


def test_cleanup_failure_is_the_final_journal_outcome(tmp_path, monkeypatch):
    import tai.fitd as implementation

    release_chats = implementation._release_chats

    def fail_cleanup(chats, owner):
        release_chats(chats, owner)
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(implementation, "_release_chats", fail_cleanup)
    target, attacker, base, evaluator, _ = participants(["Opening"], ["yes"])
    experiment = FITDExperiment(
        target, attacker, base, evaluator, artifacts_dir=tmp_path, max_calls_per_conv=1
    )
    experiment.start()
    assert experiment.wait(10)
    journal = experiment.journal
    assert journal.finished and journal.error is None
    assert journal.status() == experiment.status()
    assert journal.status().state is UserState.FAILED
    assert journal.events()[-1].data["reason"] == "RuntimeError: cleanup failed"
    metadata = json.loads(
        (tmp_path / journal.status().run_id / "metadata.json").read_text()
    )
    assert metadata["state"] == "FAILED"
    assert metadata["reason"] == journal.status().reason


def test_preflight_reports_real_exchange_and_verdict_events(tmp_path, capsys):
    target = LLMChat(lambda turns: "READY")
    attacker = LLMChat(lambda turns: "READY")
    judge = LLMChat(lambda turns: '{"success": false}')
    profile = RoleProfile(
        target, attacker, ModelEvaluatorProfile(judge, MODEL_RUBRIC_VERSION), 1
    )
    assert run_preflight(profile, tmp_path / "report.json") == 0
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
