"""Responsive controls preserve already admitted work and its outcome."""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from threading import Event

import pytest

from fitd_support import (
    RecordingEvaluator, ScriptedModel, Turns, evaluation, presented,
    read_and_assert_final_artifacts, reason, releasing, run_id,
)

from attacks.fitd.native import GarakException
from tai import (
    ChatCancelled, ContainsTextEvaluator, FITDExperiment as FITDUser,
    LLMChat, Message, UserState, capture_run, save_run,
)


@dataclass
class HeldCall:
    role: str
    index: int
    error_type: type[Exception] | None
    entered: Event
    cleanup: Event
    signals: list[Event]

    def generate(self, model: ScriptedModel, turns: Turns, cancellation: Event) -> str:
        if len(model.requests) != self.index:
            return model(turns)
        model.requests.append(turns)
        self.signals.append(cancellation)
        self.entered.set()
        assert self.cleanup.wait(5), "test did not allow model cleanup"
        assert not cancellation.is_set()
        if self.error_type is not None:
            raise self.error_type()
        return "Completed after stop"


@dataclass
class StopRun:
    user: FITDUser
    chats: dict[str, LLMChat]
    models: dict[str, ScriptedModel]
    evaluator: RecordingEvaluator
    held: HeldCall


def _stop_run(root: Path, held: HeldCall, raw_response: str) -> StopRun:
    models = {
        "Target": ScriptedModel([raw_response, "Second", "Third", "Fourth", "CONTROL"]),
        "Attacker": ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5),
    }

    def generate(turns: Turns, cancellation: Event) -> str:
        return held.generate(models[held.role], turns, cancellation)

    chats = {
        role: LLMChat.cancellable(generate, system=f"{role} instructions")
        if role == held.role else LLMChat(model, system=f"{role} instructions")
        for role, model in models.items()
    }
    evaluator = RecordingEvaluator("CONTROL")
    user = FITDUser(chats["Target"], chats["Attacker"], Message("Custom request"), evaluator, journal_dir=root)
    return StopRun(user, chats, models, evaluator, held)


@pytest.mark.parametrize(
    ("held_role", "held_index", "raw_response", "expected"),
    [("Target", 0, "First", None), ("Target", 1, "First", "First"), ("Attacker", 0, "First", None), ("Attacker", 4, "", None)],
)
@pytest.mark.parametrize("error_type", [None, RuntimeError, GarakException, ChatCancelled])
def test_stop_finishes_current_call_without_admitting_more_strategy_work(
    tmp_path: Path, held_role: str, held_index: int, raw_response: str,
    expected: str | None, error_type: type[Exception] | None,
):
    held = HeldCall(held_role, held_index, error_type, Event(), Event(), [])
    run = _stop_run(tmp_path, held, raw_response)
    assert run.user.stop() is None
    assert run.user.status().state is UserState.IDLE
    with ThreadPoolExecutor(max_workers=1) as control, releasing(run.user, held.cleanup):
        result, histories, requests = _stop_active(run, control, tmp_path, expected)
    _assert_stopped_run(run, result, histories, requests)
    _assert_stop_events(run, tmp_path, requests, raw_response)


def _stop_active(run: StopRun, control: ThreadPoolExecutor, root: Path, expected: str | None):
    user = run.user
    assert control.submit(user.start).result(timeout=1) is None
    assert run.held.entered.wait(5)
    running = control.submit(user.status).result(timeout=1)
    result = control.submit(user.result).result(timeout=1)
    assert running.state is UserState.RUNNING
    with pytest.raises(RuntimeError):
        control.submit(user.start).result(timeout=1)
    directory = root / run_id(user)
    if expected is None:
        assert result is None
        assert not (directory / "result.txt").exists()
    else:
        assert result is not None
        assert result.to_string() == expected
    assert not (directory / "result.txt").exists()
    assert not (directory / "metadata.json").exists()
    histories = {role: chat.history() for role, chat in run.chats.items()}
    requests = {role: len(model.requests) for role, model in run.models.items()}
    _assert_stop_responsive(run, control, result)
    assert {role: chat.history() for role, chat in run.chats.items()} == histories
    return result, histories, requests


def _assert_stop_responsive(run: StopRun, control: ThreadPoolExecutor, result: Message | None):
    user = run.user
    assert control.submit(user.stop).result(timeout=1) is None
    assert not run.held.signals[0].is_set()
    assert control.submit(user.stop).result(timeout=1) is None
    assert len([e for e in user.journal.events() if e.kind == "stop_requested"]) == 1
    assert control.submit(user.status).result(timeout=1).state is UserState.STOPPING
    assert not user.wait(0)
    assert control.submit(user.result).result(timeout=1) is result
    with pytest.raises(RuntimeError):
        control.submit(user.start).result(timeout=1)


def _assert_stopped_run(run: StopRun, result: Message | None, histories: dict[str, Turns], requests: dict[str, int]):
    user, error_type = run.user, run.held.error_type
    stopped = user.status()
    assert stopped.state is (UserState.STOPPED if error_type is None else UserState.FAILED), stopped.reason
    if error_type is None:
        assert stopped.reason == "user_stop"
    else:
        assert reason(user).startswith(error_type.__name__ + ":")
    assert stopped.evaluation is None
    assert user.result() is result
    assert user.stop() is None
    assert user.status() == stopped
    assert not run.evaluator.results
    assert {role: len(model.requests) for role, model in run.models.items()} == requests
    for role, chat in run.chats.items():
        if role == run.held.role and error_type is None:
            assert chat.history()[:-2] == histories[role]
            assert chat.history()[-1].message.to_string() == "Completed after stop"
        else:
            assert chat.history() == histories[role]
    assert stopped.progress.target_calls == requests["Target"] - (run.held.role == "Target" and error_type is not None)
    assert stopped.progress.attacker_calls == requests["Attacker"] - (run.held.role == "Attacker" and error_type is not None)


def _assert_stop_events(run: StopRun, root: Path, requests: dict[str, int], raw_response: str):
    events, metadata = read_and_assert_final_artifacts(run.user, root)
    assert metadata["evaluation"] is None
    starts = [e["data"] for e in events if e["kind"] == "call_started"]
    ends = [e["data"] for e in events if e["kind"] == "call_completed"]
    assert len(starts) == len(ends) + (run.held.error_type is not None) == sum(requests.values())
    stop = next(e["sequence"] for e in events if e["kind"] == "stop_requested")
    assert not any(
        e["sequence"] > stop and e["kind"] in {
            "call_started", "local_decision", "branch_selected", "result_published", "result_withdrawn",
            "evaluation_started", "history_reset", "history_restored",
        } for e in events
    )
    if run.held.error_type is None:
        assert ends[-1]["call_id"] == starts[-1]["call_id"]
        assert ends[-1]["response"] == json.loads(run.chats[run.held.role].history()[-1].message.dump())
        assert not any(e["kind"] == "call_cancelled" for e in events)
    if run.held.role == "Attacker" and run.held.index == 4:
        assert run.chats["Target"].history()[-1].message.to_string() == raw_response
        assert [end["response"]["text"] for end in ends if end["role"] == "Target"] == [raw_response]
        assert not any(event["kind"] == "result_published" for event in events)


@pytest.mark.parametrize("held_role", ["Target", "Attacker"])
def test_control_methods_remain_responsive_during_model_calls(tmp_path: Path, held_role: str):
    entered, release = Event(), Event()
    attacker_model = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)
    target_model = ScriptedModel(["First", "Second", "Third", "Fourth", "CONTROL"])

    def hold(model: ScriptedModel, index: int) -> Callable[[Turns], str]:
        def generate(turns: Turns) -> str:
            if len(model.requests) == index:
                entered.set()
                assert release.wait(10), "test did not release held generation"
            return model(turns)
        return generate

    target = LLMChat(hold(target_model, 0) if held_role == "Target" else target_model)
    attacker = LLMChat(hold(attacker_model, 4) if held_role == "Attacker" else attacker_model)
    evaluator = RecordingEvaluator("CONTROL")
    user = FITDUser(target, attacker, Message("Custom request"), evaluator, journal_dir=tmp_path)
    with ThreadPoolExecutor(max_workers=1) as control, releasing(user, release):
        started = control.submit(user.start)
        assert entered.wait(5)
        assert started.result(timeout=1) is None
        snapshot = control.submit(user.status).result(timeout=1)
        assert control.submit(user.result).result(timeout=1) is None
        assert snapshot.state is UserState.RUNNING
        assert not evaluator.results
        with pytest.raises(FrozenInstanceError):
            setattr(snapshot, "reason", "changed")
        with pytest.raises(RuntimeError):
            control.submit(user.start).result(timeout=1)
        assert user.status().run_id == snapshot.run_id
        _assert_held_history(user, target, tmp_path, held_role)
    assert user.status().state is UserState.COMPLETED
    assert snapshot.state is UserState.RUNNING


def _assert_held_history(user: FITDUser, target: LLMChat, root: Path, held_role: str):
    if held_role == "Attacker":
        directory = root / run_id(user)
        assert not (directory / "result.txt").exists()
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        completed = [e for e in events if e["kind"] == "call_completed" and e["data"]["role"] == "Target"]
        assert completed[0]["data"]["response"]["text"] == "First"
        assert target.history()[-1].message.to_string() == "First"
        assert not any(e["kind"] == "result_published" for e in events)
    else:
        assert target.history() == ()


@pytest.mark.parametrize("outcome", [True, False, GarakException("evaluation failed")])
def test_stop_during_evaluation_waits_for_execution_and_retains_its_outcome(tmp_path: Path, outcome: bool | Exception):
    entered, release = Event(), Event()

    class HoldingEvaluator:
        def evaluate(self, base: Message, result: Message) -> bool:
            entered.set()
            assert release.wait(5)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    target_model = ScriptedModel(["One", "Two", "Three", "Four", "Presented"])
    attacker_model = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)
    user = FITDUser(LLMChat(target_model), LLMChat(attacker_model), Message("Custom request"), HoldingEvaluator(), journal_dir=tmp_path)
    user.start()
    with releasing(user, release):
        assert entered.wait(5)
        result = presented(user)
        assert user.stop() is None
        assert user.status().state is UserState.STOPPING
        assert not user.wait(0)
    expected = UserState.FAILED if isinstance(outcome, Exception) else UserState.STOPPED
    assert user.status().state is expected
    assert user.result() is result
    assert len(target_model.requests) == 5
    assert len(attacker_model.requests) == 9
    _, metadata = read_and_assert_final_artifacts(user, tmp_path)
    if isinstance(outcome, Exception):
        assert metadata["evaluation"] is None
        assert "evaluation failed" in reason(user)
    else:
        assert metadata["evaluation"] == {"result_id": result.id, "success": outcome}


def test_stop_during_external_saving_is_responsive_and_preserves_the_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    entered, release = Event(), Event()
    write_text = Path.write_text

    def hold_metadata(path: Path, data: str, encoding: str | None = None, errors: str | None = None, newline: str | None = None) -> int:
        if path.name == "metadata.json.tmp" and not entered.is_set():
            entered.set()
            assert release.wait(5), "test did not release artifact write"
        return write_text.__call__(path, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", hold_metadata)
    target = ScriptedModel(["One", "Two", "Three", "Four", "CONTROL"])
    attacker = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)
    user = FITDUser(LLMChat(target), LLMChat(attacker), Message("Custom request"), ContainsTextEvaluator("CONTROL"), journal_dir=tmp_path)
    user.start()
    assert user.wait(10)
    record = capture_run(user, journal=user.journal)
    with ThreadPoolExecutor(max_workers=1) as control:
        saving = control.submit(save_run, record, tmp_path)
        with releasing(user, release):
            assert entered.wait(5)
            assert user.status().state is UserState.COMPLETED
            assert user.stop() is None
            assert user.snapshot() == record.snapshot
            assert user.wait(0)
        saving.result(timeout=5)
    _assert_saved_outcome(user, tmp_path, target, attacker)


def _assert_saved_outcome(user: FITDUser, root: Path, target: ScriptedModel, attacker: ScriptedModel):
    assert user.status().state is UserState.COMPLETED
    assert user.status().reason == "evaluator_success"
    assert evaluation(user).success is True
    assert len(target.requests) == 5
    assert len(attacker.requests) == 9
    events, metadata = read_and_assert_final_artifacts(user, root)
    finishes = [e["data"] for e in events if e["kind"] == "run_finished"]
    assert len(finishes) == 1
    assert metadata["state"] == "COMPLETED"
