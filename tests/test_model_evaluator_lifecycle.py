"""Real FITD assessment admission/completion and ownership of the third role."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from evaluation_fakes import HeldResponse, ScriptedModel
from journal_assertions import (
    at, evaluation_of, mapping, released_worker, result_of, run_directory, sequence, text,
)
from runs_artifacts import read_and_assert_final_artifacts
from test_model_evaluator import make_user

from tai import (
    EvaluatorSetup,
    ChatTurn,
    LLMChat,
    Message,
    RunJournal,
    UserState,
)
from tai import (
    FITDExperiment as FITDUser,
)


def assert_active_assessment_stop(
    user: FITDUser, judge: LLMChat, held: HeldResponse, control: ThreadPoolExecutor,
) -> tuple[Message, tuple[ChatTurn, ...]]:
    assert held.entered.wait(5)
    result = result_of(user)
    assert user.status().evaluation is None
    assert control.submit(user.stop).result(timeout=1) is None
    assert held.cancellation is not None and not held.cancellation.is_set()
    assert control.submit(user.status).result(timeout=1).state is UserState.STOPPING
    assert control.submit(user.result).result(timeout=1) is result
    assert not user.wait(0)
    assert control.submit(user.stop).result(timeout=1) is None
    with pytest.raises(RuntimeError, match="active run"):
        user.start()
    history = judge.history()
    assert len(history) == 1  # The previous assessment was reset.
    return result, history


def assert_stopped_artifacts(user: FITDUser, root: Path, result: Message, success: bool) -> None:
    events, metadata = read_and_assert_final_artifacts(user, root)
    assert metadata["evaluation"] == {"result_id": result.id, "success": success}
    completed = [mapping(e["data"]) for e in events if e["kind"] == "evaluation_completed"]
    assert len(completed) == 2 and completed[0]["success"] is False
    assert completed[0]["result_id"] != result.id
    assert {k: completed[-1][k] for k in ("result_id", "success")} == metadata["evaluation"]
    assert not any(text(e["kind"]).endswith("_cancelled") for e in events)
    stop = next(e["sequence"] for e in events if e["kind"] == "stop_requested")
    assert isinstance(stop, int)
    assert not any(
        e["kind"] == "call_started" and sequence_number(e) > stop for e in events
    )


def sequence_number(event: dict[str, object]) -> int:
    value = event["sequence"]
    assert isinstance(value, int)
    return value


@pytest.mark.parametrize("outcome", ['{"success": true}', '{"success": false}'])
def test_stop_during_model_assessment_keeps_completed_work_and_waits_for_cleanup(
    tmp_path: Path, outcome: str
):
    held = HeldResponse(outcome)
    model = ScriptedModel(['{"success": false}', held])
    judge = LLMChat.cancellable(model, system="Evaluator system")
    user, target, attacker, _ = make_user(tmp_path, judge)
    user.start()
    with ThreadPoolExecutor(max_workers=1) as control:
        with released_worker(user, held.release):
            result, history = assert_active_assessment_stop(user, judge, held, control)
            requests = len(target.requests), len(attacker.requests), len(model.requests)
    assert user.status().state is UserState.STOPPED, user.status().reason
    success = json.loads(outcome)["success"]
    assert evaluation_of(user).result_id == result.id
    assert evaluation_of(user).success is success
    assert user.status().progress.evaluator_calls == 2
    assert user.result() is result and judge.history()[:-2] == history
    assert judge.history()[-1].message.to_string() == outcome
    assert (
        len(target.requests),
        len(attacker.requests),
        len(model.requests),
    ) == requests
    assert_stopped_artifacts(user, tmp_path, result, success)
    # Reusing the same chat in a new run gets an independent signal.
    model.responses.append('{"success": true}')
    next_user, _, _, _ = make_user(tmp_path, judge)
    next_user.start()
    assert next_user.wait(10)
    assert next_user.status().state is UserState.COMPLETED
    assert model.signals[-1] is not held.cancellation
    assert not model.signals[-1].is_set()
    assert len(model.requests[-1]) == 2


@pytest.mark.parametrize("boundary", ["evaluation_started", "call_started"])
def test_stop_between_assessment_admission_and_send_finishes_whole_assessment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
):
    entered, release = Event(), Event()
    requests: list[tuple[ChatTurn, ...]] = []
    judge = LLMChat(lambda turns: requests.append(tuple(turns)) or '{"success": true}')
    user, target, attacker, _ = make_user(tmp_path, judge)
    flush = RunJournal.flush

    def held_flush(execution: RunJournal) -> None:
        flush(execution)
        last = execution.events()[-1]
        if (
            last.kind == boundary
            and (boundary == "evaluation_started" or at(last.data, "role") == "Evaluator")
            and not entered.is_set()
        ):
            entered.set()
            assert release.wait(5)

    monkeypatch.setattr(RunJournal, "flush", held_flush)
    user.start()
    with released_worker(user, release):
        assert entered.wait(5)
        user.stop()
        assert user.status().state is UserState.STOPPING
    assert user.status().state is UserState.STOPPED
    assert evaluation_of(user).success is True
    assert evaluation_of(user).result_id == result_of(user).id
    assert len(requests) == 1
    assert len(target.requests) == 5 and len(attacker.requests) == 9
    events, _ = read_and_assert_final_artifacts(user, tmp_path)
    stop = next(sequence_number(e) for e in events if e["kind"] == "stop_requested")
    later_calls = [
        mapping(e["data"])
        for e in events
        if e["kind"] == "call_started" and sequence_number(e) > stop
    ]
    assert len(later_calls) == (boundary == "evaluation_started")
    assert all(e["role"] == "Evaluator" for e in later_calls)
    assert not any(
        sequence_number(e) > stop and e["kind"] in {"evaluation_started", "branch_selected"}
        for e in events
    )


@pytest.mark.parametrize("terminal", ["completed", "stopped", "failed"])
def test_evaluator_chat_is_reserved_atomically_until_terminal_cleanup(
    tmp_path: Path, terminal: str
):
    held = HeldResponse(
        RuntimeError("Evaluator failed")
        if terminal == "failed"
        else '{"success": true}'
    )
    model = ScriptedModel([held, '{"success": true}'])
    judge = LLMChat.cancellable(model)
    owner, _, _, _ = make_user(tmp_path, judge)
    contender, _, _, _ = make_user(tmp_path, judge)
    owner.start()
    with released_worker(owner, held.release):
        assert held.entered.wait(5)
        histories = [
            chat.history() for chat in (*contender._settings.chats.values(),)
        ]
        with pytest.raises(RuntimeError, match="LLMChat.*active"):
            contender.start()
        assert contender.status().state is UserState.IDLE
        assert contender.journal.events() == ()
        assert [
            chat.history() for chat in (*contender._settings.chats.values(),)
        ] == histories
        assert all(
            chat._execution.owner is None
            for role, chat in contender._settings.chats.items() if role != "Evaluator"
        )
        if terminal == "stopped":
            owner.stop()
            with pytest.raises(RuntimeError, match="LLMChat.*active"):
                contender.start()
    assert (
        owner.status().state
        is {
            "completed": UserState.COMPLETED,
            "stopped": UserState.STOPPED,
            "failed": UserState.FAILED,
        }[terminal]
    )
    events = owner.journal.events()
    contender.start()
    assert contender.wait(10)
    assert contender.status().state is UserState.COMPLETED, contender.status().reason
    assert owner.journal.events() == events
    assert len(model.requests[-1]) == 1


def test_restarting_same_user_resets_evaluator_and_preserves_prior_artifacts(tmp_path: Path):
    target_model = ScriptedModel(["One", "Two", "Three", "Four", "Presented"] * 2)
    attacker_model = ScriptedModel((["Initial", "L1", "L2", "L3"] + ["yes"] * 5) * 2)
    model = ScriptedModel(['{"success": true}'] * 2)
    judge = LLMChat.cancellable(model, system="Evaluator system")
    base = Message("Base")
    user = FITDUser(
        LLMChat.cancellable(target_model),
        LLMChat.cancellable(attacker_model),
        base,
        EvaluatorSetup.model(judge),
        journal_dir=tmp_path,
    )
    user.start()
    assert user.wait(10)
    old_status = user.status()
    old_id = judge.conversation_id
    directory = run_directory(user, tmp_path)
    old_files = {p.name: p.read_bytes() for p in directory.iterdir()}
    old_checkpoint = judge.checkpoint()
    user.start()
    assert user.wait(10)
    assert user.status().state is UserState.COMPLETED
    assert user.status().run_id != old_status.run_id
    assert judge.conversation_id != old_id
    assert len(model.requests) == 2 and all(len(r) == 2 for r in model.requests)
    assert model.signals[0] is not model.signals[1]
    with pytest.raises(ValueError, match="stale"):
        judge.restore(old_checkpoint)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == old_files
    _, metadata = read_and_assert_final_artifacts(user, tmp_path)
    assert at(metadata, "settings", "base") == json.loads(base.dump())
    assert len(sequence(at(metadata, "settings", "Evaluator", "initial_history"))) == 1


@pytest.mark.parametrize("success", [True, False])
def test_stop_after_completed_evaluator_exchange_retains_its_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, success: bool
):
    entered, release = Event(), Event()
    judge = LLMChat(lambda turns: json.dumps({"success": success}))
    user, target, attacker, _ = make_user(tmp_path, judge)
    emit = RunJournal.record

    def hold_completed(execution: RunJournal, kind: str, **data: object) -> None:
        emit(execution, kind, **data)
        if kind == "call_completed" and data["role"] == "Evaluator":
            entered.set()
            assert release.wait(5)

    monkeypatch.setattr(RunJournal, "record", hold_completed)
    user.start()
    with released_worker(user, release):
        assert entered.wait(5)
        history = judge.history()
        assert json.loads(history[-1].message.to_string())["success"] is success
        user.stop()
        assert user.status().state is UserState.STOPPING
    assert user.status().state is UserState.STOPPED
    assert evaluation_of(user).success is success
    assert evaluation_of(user).result_id == result_of(user).id
    assert user.status().progress.evaluator_calls == 1
    assert judge.history() == history
    assert len(target.requests) == 5 and len(attacker.requests) == 9
    events, metadata = read_and_assert_final_artifacts(user, tmp_path)
    assert at(metadata, "evaluation", "success") is success
    assert not any(e["kind"] == "call_cancelled" for e in events)


@pytest.mark.parametrize(
    "outcome", [RuntimeError("model failed"), '{"success": "false"}']
)
def test_stop_does_not_mask_model_assessment_failure(tmp_path: Path, outcome: str | Exception):
    held = HeldResponse(outcome)
    judge = LLMChat.cancellable(ScriptedModel([held]))
    user, target, attacker, _ = make_user(tmp_path, judge)
    user.start()
    with released_worker(user, held.release):
        assert held.entered.wait(5)
        result = result_of(user)
        user.stop()
        assert held.cancellation is not None and not held.cancellation.is_set()
    assert user.status().state is UserState.FAILED
    assert user.result() is result and user.status().evaluation is None
    assert len(target.requests) == 5 and len(attacker.requests) == 9
    events, metadata = read_and_assert_final_artifacts(user, tmp_path)
    assert metadata["evaluation"] is None
    assert any(
        e["kind"] == "evaluation_failed" and at(e, "data", "result_id") == result.id
        for e in events
    )
    expected = (
        "ModelEvaluationError"
        if isinstance(outcome, Exception)
        else "ModelVerdictError"
    )
    assert expected in text(user.status().reason)
