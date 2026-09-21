"""Stop at real operation boundaries, including worker-side journal writes."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from test_fitd import RecordingEvaluator, ScriptedModel, read_and_assert_final_artifacts

from tai import FITDExperiment as FITDUser
from tai import LLMChat, Message, RunJournal, UserState


@pytest.mark.parametrize(
    ("boundary", "role", "occurrence", "target_calls", "attacker_calls", "presented"),
    [
        ("call_started", "Attacker", 1, 0, 1, None),
        ("call_started", "Target", 1, 1, 4, None),
        ("call_completed", "Target", 1, 1, 4, None),
        ("target_exchange", None, 1, 1, 4, None),
        ("branch_selected", None, 1, 1, 5, None),
        ("result_published", None, 5, 5, 9, "Presented"),
        ("evaluation_completed", None, 1, 5, 9, "Presented"),
    ],
)
def test_stop_at_admission_or_return_prevents_the_next_action(
    tmp_path,
    monkeypatch,
    boundary,
    role,
    occurrence,
    target_calls,
    attacker_calls,
    presented,
):
    entered, release = Event(), Event()
    flush = RunJournal.flush

    def held_flush(execution):
        flush(execution)
        events = execution.events()
        matches = [
            e
            for e in events
            if e.kind == boundary and (role is None or e.data.get("role") == role)
        ]
        if (
            len(matches) == occurrence
            and matches[-1] is events[-1]
            and not entered.is_set()
        ):
            entered.set()
            assert release.wait(5)

    monkeypatch.setattr(RunJournal, "flush", held_flush)
    target = ScriptedModel(["One", "Two", "Three", "Four", "Presented"])
    attacker = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)
    evaluator = RecordingEvaluator("CONTROL")
    target_chat = LLMChat(target)
    user = FITDUser(
        target_chat,
        LLMChat(attacker),
        Message("Base"),
        evaluator,
        artifacts_dir=tmp_path,
    )
    user.start()
    with ThreadPoolExecutor(max_workers=1) as control:
        try:
            assert entered.wait(5)
            assert control.submit(user.stop).result(timeout=1) is None
            user.stop()
            assert user.status().state is UserState.STOPPING
            assert not user.wait(0)
        finally:
            release.set()
            assert user.wait(10)
    assert user.status().state is UserState.STOPPED, user.status().reason
    assert len(target.requests) == user.status().progress.target_calls == target_calls
    assert (
        len(attacker.requests)
        == user.status().progress.attacker_calls
        == attacker_calls
    )
    result = user.result()
    assert (result.to_string() if result is not None else None) == presented
    assert len(evaluator.results) == (boundary == "evaluation_completed")
    if boundary == "evaluation_completed":
        assert user.status().evaluation.result_id == result.id
        assert user.status().evaluation.success is False
    else:
        assert user.status().evaluation is None
    events, _ = read_and_assert_final_artifacts(user, tmp_path)
    stop = next(e["sequence"] for e in events if e["kind"] == "stop_requested")
    assert not any(
        e["sequence"] > stop
        and e["kind"]
        in {
            "call_started",
            "evaluation_started",
            "branch_selected",
            "local_decision",
            "result_published",
            "result_withdrawn",
            "history_reset",
            "history_restored",
        }
        for e in events
    )
    if target_calls:
        returns = [
            e["data"]["response"]
            for e in events
            if e["kind"] == "call_completed" and e["data"]["role"] == "Target"
        ]
        assert returns[-1]["id"] == target_chat.history()[-1].message.id
        points = [e for e in events if e["kind"] == "target_exchange"]
        assert points[-1]["data"]["response_id"] == returns[-1]["id"]
