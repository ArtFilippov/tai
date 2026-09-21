import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event

import pytest
from garak.exception import GarakException

from tai import (
    ChatCancelled,
    ContainsTextEvaluator,
    Evaluator,
    FITDUser,
    LLMChat,
    Message,
    UserState,
)


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def __call__(self, turns):
        self.requests.append(turns)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingEvaluator(Evaluator):
    def __init__(self, criterion):
        self.evaluator = ContainsTextEvaluator(criterion)
        self.results = []

    def is_success(self, result):
        self.results.append(result)
        return self.evaluator.is_success(result)


def read_and_assert_final_artifacts(user, root):
    run_dir = root / user.status().run_id
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert events == [json.loads(event.dump()) for event in user.events()]
    assert events[-1]["kind"] == "run_finished"
    assert metadata["state"] == user.status().state.value
    assert events[-1]["data"]["state"] == metadata["state"]
    assert events[-1]["data"]["revision"] == metadata["finalization_revision"]
    assert events[-1]["data"]["reason"] == metadata["reason"] == user.status().reason
    result = user.result()
    if result is None:
        assert metadata["result"] is None
        assert metadata["result_id"] is None
        assert not (run_dir / "result.txt").exists()
    else:
        assert metadata["result"] == json.loads(result.dump())
        assert metadata["result_id"] == result.id
        assert (run_dir / "result.txt").read_text() == result.to_string()
    return events, metadata


@pytest.mark.parametrize(
    ("held_role", "held_index", "raw_response", "presented"),
    [
        ("Target", 0, "First", None),
        ("Target", 1, "First", "First"),
        ("Attacker", 0, "First", None),
        ("Attacker", 4, "", None),
    ],
)
@pytest.mark.parametrize(
    "error_type", [None, RuntimeError, GarakException, ChatCancelled]
)
def test_stop_finishes_current_call_without_admitting_more_strategy_work(
    tmp_path, held_role, held_index, raw_response, presented, error_type
):
    entered, cleanup = Event(), Event()
    signals = []
    models = {
        "Target": ScriptedModel([raw_response, "Second", "Third", "Fourth", "CONTROL"]),
        "Attacker": ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5),
    }

    def held_generate(turns, cancellation):
        model = models[held_role]
        if len(model.requests) != held_index:
            return model(turns)
        model.requests.append(turns)
        signals.append(cancellation)
        entered.set()
        assert cleanup.wait(5), "test did not allow model cleanup"
        assert not cancellation.is_set()
        if error_type is not None:
            raise error_type()
        return "Completed after stop"

    chats = {
        role: LLMChat.cancellable(held_generate, system=f"{role} instructions")
        if role == held_role
        else LLMChat(model, system=f"{role} instructions")
        for role, model in models.items()
    }
    evaluator = RecordingEvaluator("CONTROL")
    user = FITDUser(
        chats["Target"],
        chats["Attacker"],
        Message("Custom request"),
        evaluator,
        artifacts_dir=tmp_path,
    )
    assert user.stop() is None
    assert user.status().state is UserState.IDLE
    with ThreadPoolExecutor(max_workers=1) as control:
        try:
            assert control.submit(user.start).result(timeout=1) is None
            assert entered.wait(5)
            running = control.submit(user.status).result(timeout=1)
            result = control.submit(user.result).result(timeout=1)
            assert running.state is UserState.RUNNING
            with pytest.raises(RuntimeError):
                control.submit(user.start).result(timeout=1)
            if presented is None:
                assert result is None
                assert not (tmp_path / running.run_id / "result.txt").exists()
            else:
                assert result.to_string() == presented
                assert (
                    tmp_path / running.run_id / "result.txt"
                ).read_text() == presented
            histories = {role: chat.history() for role, chat in chats.items()}
            requests = {role: len(model.requests) for role, model in models.items()}
            assert control.submit(user.stop).result(timeout=1) is None
            assert not signals[0].is_set()
            assert control.submit(user.stop).result(timeout=1) is None
            assert len([e for e in user.events() if e.kind == "stop_requested"]) == 1
            assert (
                control.submit(user.status).result(timeout=1).state
                is UserState.STOPPING
            )
            assert not user.wait(0)
            assert control.submit(user.result).result(timeout=1) is result
            with pytest.raises(RuntimeError):
                control.submit(user.start).result(timeout=1)
            assert {role: chat.history() for role, chat in chats.items()} == histories
        finally:
            cleanup.set()
            assert user.wait(10)
    stopped = user.status()
    assert stopped.state is (
        UserState.STOPPED if error_type is None else UserState.FAILED
    ), stopped.reason
    if error_type is None:
        assert stopped.reason == "user_stop"
    else:
        assert stopped.reason.startswith(error_type.__name__ + ":")
    assert stopped.evaluation is None
    assert user.result() is result
    assert user.stop() is None
    assert user.status() == stopped
    assert evaluator.results == []
    assert {role: len(model.requests) for role, model in models.items()} == requests
    for role, chat in chats.items():
        if role == held_role and error_type is None:
            assert chat.history()[:-2] == histories[role]
            assert chat.history()[-1].message.to_string() == "Completed after stop"
        else:
            assert chat.history() == histories[role]
    events, metadata = read_and_assert_final_artifacts(user, tmp_path)
    assert metadata["evaluation"] is None
    starts = [e["data"] for e in events if e["kind"] == "call_started"]
    ends = [e["data"] for e in events if e["kind"] == "call_completed"]
    assert len(starts) == len(ends) + (error_type is not None) == sum(requests.values())
    stop = next(e["sequence"] for e in events if e["kind"] == "stop_requested")
    assert not any(
        e["sequence"] > stop
        and e["kind"]
        in {
            "call_started",
            "local_decision",
            "branch_selected",
            "result_published",
            "result_withdrawn",
            "evaluation_started",
            "history_reset",
            "history_restored",
        }
        for e in events
    )
    if error_type is None:
        assert ends[-1]["call_id"] == starts[-1]["call_id"]
        assert ends[-1]["response"] == json.loads(
            chats[held_role].history()[-1].message.dump()
        )
        assert not any(e["kind"] == "call_cancelled" for e in events)
    assert stopped.progress.target_calls == requests["Target"] - (
        held_role == "Target" and error_type is not None
    )
    assert stopped.progress.attacker_calls == requests["Attacker"] - (
        held_role == "Attacker" and error_type is not None
    )
    if held_role == "Attacker" and held_index == 4:
        assert chats["Target"].history()[-1].message.to_string() == raw_response
        assert [end["response"]["text"] for end in ends if end["role"] == "Target"] == [
            raw_response
        ]
        assert not any(event["kind"] == "result_published" for event in events)


def test_real_fitd_uses_the_supplied_request_and_publishes_the_evaluated_result(
    tmp_path,
):
    # Initial safe request, three ladder levels, then alignment decisions.
    attacker_model = ScriptedModel(
        ["Describe a fictional puzzle", "Level one", "Level two", "Level three"]
        + ["yes"] * 5
    )
    target_model = ScriptedModel(
        [
            "Puzzle introduction",
            "More detail",
            "Further detail",
            "Almost done",
            "CONTROL",
        ]
    )
    target = LLMChat(target_model, model="same-model", system="Target instructions")
    attacker = LLMChat(
        attacker_model, model="same-model", system="Attacker instructions"
    )
    evaluator = RecordingEvaluator("CONTROL")
    base = Message("Explain the final fictional puzzle solution")
    user = FITDUser(target, attacker, base, evaluator, artifacts_dir=tmp_path)

    assert user.status().state is UserState.IDLE
    assert user.result() is None
    assert user.start() is None
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.COMPLETED, status.reason
    assert status.reason == "evaluator_success"
    assert status.progress.target_calls == 5
    assert base.to_string() in attacker_model.requests[0][-1].message.to_string()
    assert (
        target_model.requests[0][-1].message.to_string()
        == "Describe a fictional puzzle"
    )
    assert target_model.requests[-1][-1].message.to_string() == base.to_string()
    assert len(target.history()) == 11
    assert all(
        r[0].message.to_string() == "Attacker instructions"
        for r in attacker_model.requests
    )
    result = user.result()
    assert result is not None
    assert "CONTROL" in result.to_string()
    assert evaluator.results == [result]
    assert evaluator.results[0] is result
    assert status.evaluation.result_id == result.id
    assert status.evaluation.success is True

    run_dir = tmp_path / status.run_id
    assert (run_dir / "result.txt").read_text() == result.to_string()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["state"] == "COMPLETED"
    assert metadata["result"]["id"] == result.id
    assert metadata["evaluation"] == {"result_id": result.id, "success": True}
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert [event.kind for event in user.events()] == [e["kind"] for e in events]
    assert all(e["run_id"] == status.run_id for e in events)
    assert events[0]["kind"] == "run_started"
    assert events[-1]["kind"] == "run_finished"
    starts = {e["data"]["call_id"]: e for e in events if e["kind"] == "call_started"}
    ends = [e for e in events if e["kind"] == "call_completed"]
    assert len(starts) == len(ends) == 14
    assert {e["data"]["role"] for e in ends} == {"Target", "Attacker"}
    assert all(e["data"]["call_id"] in starts for e in ends)
    assert user.stop() is None
    assert user.status() == status
    assert user.result() is result


@pytest.mark.parametrize("held_role", ["Target", "Attacker"])
def test_control_methods_remain_responsive_during_model_calls(tmp_path, held_role):
    entered, release = Event(), Event()
    attacker_model = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)
    target_model = ScriptedModel(["First", "Second", "Third", "Fourth", "CONTROL"])

    def hold(model, index):
        def generate(turns):
            if len(model.requests) == index:
                entered.set()
                assert release.wait(10), "test did not release held generation"
            return model(turns)

        return generate

    target = LLMChat(hold(target_model, 0) if held_role == "Target" else target_model)
    # Hold the internal alignment check after the first completed Target exchange.
    attacker = LLMChat(
        hold(attacker_model, 4) if held_role == "Attacker" else attacker_model
    )
    evaluator = RecordingEvaluator("CONTROL")
    user = FITDUser(
        target, attacker, Message("Custom request"), evaluator, artifacts_dir=tmp_path
    )
    with ThreadPoolExecutor(max_workers=1) as control:
        try:
            started = control.submit(user.start)
            assert entered.wait(5)
            assert started.result(timeout=1) is None
            snapshot = control.submit(user.status).result(timeout=1)
            result = control.submit(user.result).result(timeout=1)
            assert snapshot.state is UserState.RUNNING
            assert evaluator.results == []
            with pytest.raises(FrozenInstanceError):
                snapshot.reason = "changed"
            with pytest.raises(RuntimeError):
                control.submit(user.start).result(timeout=1)
            assert user.status().run_id == snapshot.run_id
            if held_role == "Attacker":
                assert result is None
                assert not (tmp_path / snapshot.run_id / "result.txt").exists()
                events = [
                    json.loads(line)
                    for line in (tmp_path / snapshot.run_id / "events.jsonl")
                    .read_text()
                    .splitlines()
                ]
                completed = [
                    e
                    for e in events
                    if e["kind"] == "call_completed" and e["data"]["role"] == "Target"
                ]
                assert completed[0]["data"]["response"]["text"] == "First"
                assert target.history()[-1].message.to_string() == "First"
                assert not any(e["kind"] == "result_published" for e in events)
            else:
                assert result is None
                assert target.history() == ()
        finally:
            release.set()
            assert user.wait(10)
    assert user.status().state is UserState.COMPLETED
    assert snapshot.state is UserState.RUNNING


def test_one_chat_cannot_serve_both_roles(tmp_path):
    chat = LLMChat(lambda turns: "response")
    with pytest.raises(ValueError, match="independent"):
        FITDUser(
            chat,
            chat,
            Message("request"),
            ContainsTextEvaluator("CONTROL"),
            artifacts_dir=tmp_path,
        )


def test_negative_evaluation_continues_and_is_not_attributed_to_new_result(tmp_path):
    entered, release = Event(), Event()

    class HoldingEvaluator(RecordingEvaluator):
        def is_success(self, result):
            if self.results:
                entered.set()
                assert release.wait(10)
            return super().is_success(result)

    evaluator = HoldingEvaluator("CONTROL")
    attacker = LLMChat(ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 6))
    target = LLMChat(
        ScriptedModel(["One", "Two", "Three", "Four", "Not yet", "CONTROL"])
    )
    user = FITDUser(
        target, attacker, Message("Custom request"), evaluator, artifacts_dir=tmp_path
    )
    user.start()
    try:
        assert entered.wait(5)
        old_result = evaluator.results[0]
        current = user.result()
        assert old_result.to_string() == "Not yet"
        assert current.to_string() == "CONTROL"
        assert current.id != old_result.id
        assert user.status().evaluation is None
        assert user.status().result_id == current.id
        decisions = [e.data for e in user.events() if e.kind == "evaluation_completed"]
        assert decisions == [{"result_id": old_result.id, "success": False}]
        assert user.result() is current
        assert evaluator.results == [old_result]
    finally:
        release.set()
        assert user.wait(10)
    assert user.status().state is UserState.COMPLETED
    assert user.status().evaluation.success is True
    assert old_result.to_string() == "Not yet"


def test_round_limit_is_normal_completion_and_empty_result_is_present(tmp_path):
    evaluator = RecordingEvaluator("CONTROL")
    # Empty text is a completed response, distinct from a missing result.
    target = LLMChat(ScriptedModel([""] * 10))
    attacker = LLMChat(ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 10))
    user = FITDUser(
        target, attacker, Message("Custom request"), evaluator, artifacts_dir=tmp_path
    )
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.COMPLETED, status.reason
    assert status.reason == "round_limit"
    assert status.progress.target_calls == 10
    assert status.evaluation.success is False
    # Only adversarial-level results are evaluated, rather than each exchange.
    assert len(evaluator.results) == 6
    assert user.result() is evaluator.results[-1]
    run_dir = tmp_path / status.run_id
    assert (run_dir / "result.txt").read_text() == ""
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["result"] is not None
    assert metadata["result"]["text"] == ""
    assert metadata["evaluation"]["result_id"] == metadata["result"]["id"]
    assert len([e for e in user.events() if e.kind == "evaluation_completed"]) == 6


def test_fitd_realigns_a_response_before_continuing_the_ladder(tmp_path):
    attacker = LLMChat(
        ScriptedModel(
            ["Describe a fictional puzzle", "L1", "L2", "L3", "no"] + ["yes"] * 5
        )
    )
    target_model = ScriptedModel(
        [
            "An unrelated answer",
            "Puzzle introduction",
            "Detail",
            "More",
            "Almost",
            "CONTROL",
        ]
    )
    target = LLMChat(target_model)
    user = FITDUser(
        target,
        attacker,
        Message("Custom request"),
        ContainsTextEvaluator("CONTROL"),
        artifacts_dir=tmp_path,
    )
    user.start()
    assert user.wait(10)
    assert user.status().state is UserState.COMPLETED, user.status().reason
    correction_request = target_model.requests[1][-1].message.to_string()
    assert "An unrelated answer" in correction_request
    assert "Describe a fictional puzzle" in correction_request
    assert target_model.requests[1][-2].message.to_string() == "An unrelated answer"
    assert user.status().progress.target_calls == 6
    assert user.status().evaluation.success is True


def test_run_without_a_presented_result_has_no_result_file_or_evaluation(tmp_path):
    def unavailable(turns):
        raise RuntimeError("model unavailable")

    user = FITDUser(
        LLMChat(lambda turns: "unused"),
        LLMChat(unavailable),
        Message("request"),
        ContainsTextEvaluator("CONTROL"),
        artifacts_dir=tmp_path,
    )
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.FAILED
    assert user.result() is None
    assert status.evaluation is None
    run_dir = tmp_path / status.run_id
    assert not (run_dir / "result.txt").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["result"] is None
    assert metadata["evaluation"] is None


@pytest.mark.parametrize(
    "error_type", [RuntimeError, GarakException, NotImplementedError]
)
def test_required_evaluation_failure_preserves_new_result_without_an_old_score(
    tmp_path, error_type
):
    class FailingEvaluator(RecordingEvaluator):
        def is_success(self, result):
            if self.results:
                # The new text and all completed calls must already be durable.
                run_dir = tmp_path / user.status().run_id
                assert (run_dir / "result.txt").read_text() == "Unchecked"
                raise error_type("required evaluation unavailable")
            return super().is_success(result)

    evaluator = FailingEvaluator("CONTROL")
    target_model = ScriptedModel(
        ["One", "Two", "Three", "Four", "Negative", "Unchecked"]
    )
    attacker_model = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 6)
    user = FITDUser(
        LLMChat(target_model),
        LLMChat(attacker_model),
        Message("Custom request"),
        evaluator,
        artifacts_dir=tmp_path,
    )
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.FAILED
    assert "required evaluation unavailable" in status.reason
    assert user.result().to_string() == "Unchecked"
    assert status.evaluation is None
    assert status.progress.target_calls == 6
    assert status.progress.attacker_calls == 10
    events, metadata = read_and_assert_final_artifacts(user, tmp_path)
    assert metadata["evaluation"] is None
    evaluations = [e["data"] for e in events if e["kind"] == "evaluation_completed"]
    assert evaluations == [{"result_id": evaluator.results[0].id, "success": False}]
    failures = [e["data"] for e in events if e["kind"] == "evaluation_failed"]
    assert len(failures) == 1
    assert failures[0]["result_id"] == user.result().id
    assert (
        failures[0]["error"]
        == f"{error_type.__name__}: required evaluation unavailable"
    )
    assert user.stop() is None
    assert user.status() == status


@pytest.mark.parametrize("error_type", [RuntimeError, GarakException])
@pytest.mark.parametrize(
    ("role", "prior_target_responses", "attacker_responses"),
    [
        ("Target", [], ["Initial", "L1", "L2", "L3"]),
        ("Target", ["First"], ["Initial", "L1", "L2", "L3", "yes"]),
        ("Attacker", [], []),
        ("Attacker", [""], ["Initial", "L1", "L2", "L3"]),
    ],
)
def test_provider_failure_keeps_completed_exchanges_and_blocks_further_calls(
    tmp_path, error_type, role, prior_target_responses, attacker_responses
):
    responses = {"Target": prior_target_responses, "Attacker": attacker_responses}
    models = {
        name: ScriptedModel(
            values + ([error_type("provider failed")] if name == role else [])
        )
        for name, values in responses.items()
    }
    chats = {name: LLMChat(model, system=name) for name, model in models.items()}
    user = FITDUser(
        chats["Target"],
        chats["Attacker"],
        Message("Custom request"),
        ContainsTextEvaluator("CONTROL"),
        artifacts_dir=tmp_path,
    )
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.FAILED
    assert status.reason == f"{error_type.__name__}: provider failed"
    for name, chat in chats.items():
        retained = len(responses[name]) if name == "Target" else int(role != "Attacker")
        assert len(chat.history()) == 1 + 2 * retained
        assert len(models[name].requests) == len(responses[name]) + (name == role)
    if role == "Attacker":
        # A completed Target answer awaiting alignment is still unclassified.
        assert user.result() is None
    elif prior_target_responses:
        assert user.result().to_string() == prior_target_responses[-1]
    events, metadata = read_and_assert_final_artifacts(user, tmp_path)
    assert metadata["evaluation"] is None
    assert status.progress.target_calls == len(prior_target_responses)
    assert status.progress.attacker_calls == len(attacker_responses)
    failures = [e["data"] for e in events if e["kind"] == "call_failed"]
    assert len(failures) == 1
    assert failures[0]["role"] == role
    assert failures[0]["error"] == status.reason
    starts = [e["data"]["call_id"] for e in events if e["kind"] == "call_started"]
    assert failures[0]["call_id"] == starts[-1]
    assert len([e for e in events if e["kind"] == "call_completed"]) == len(starts) - 1
    assert user.stop() is None
    assert user.status() == status


def test_failure_after_native_restore_preserves_accepted_result_and_discarded_journal(
    tmp_path,
):
    rejected = "I cannot help with that: CONTROL"
    target_model = ScriptedModel(
        ["Retained opening", rejected, RuntimeError("provider failed after restore")]
    )
    attacker_model = ScriptedModel(
        ["Initial", "L1", "L2", "L3", "yes", "Safer request"]
    )
    target = LLMChat(target_model)
    user = FITDUser(
        target,
        LLMChat(attacker_model),
        Message("Custom request"),
        ContainsTextEvaluator("CONTROL"),
        artifacts_dir=tmp_path,
    )
    user.start()
    assert user.wait(10)
    assert user.status().state is UserState.FAILED
    assert user.status().reason == "RuntimeError: provider failed after restore"
    assert user.result().to_string() == "Retained opening"
    assert len(target_model.requests) == 3
    assert len(attacker_model.requests) == 6
    assert target_model.requests[2][-1].message.to_string() == "Safer request"
    assert target_model.requests[2][:-1] == target_model.requests[1][:-1]
    assert len(target.history()) == 2
    assert target.history()[-1].message is user.result()
    assert all(rejected not in turn.message.to_string() for turn in target.history())
    events, metadata = read_and_assert_final_artifacts(user, tmp_path)
    assert metadata["evaluation"] is None
    assert [
        e["data"]["result"]["text"] for e in events if e["kind"] == "result_published"
    ] == ["Retained opening"]
    assert any(
        e["kind"] == "call_completed" and e["data"]["response"]["text"] == rejected
        for e in events
    )
    branch = next(
        e["data"]
        for e in events
        if e["kind"] == "branch_selected" and e["data"]["branch"] == "rephrase"
    )
    assert branch["accepted"] is False
    assert any(
        e["kind"] == "history_restored" and e["data"]["cause_id"] == branch["branch_id"]
        for e in events
    )
    assert not any(e["kind"] == "evaluation_completed" for e in events)


@pytest.mark.parametrize("outcome", [True, False, GarakException("evaluation failed")])
def test_stop_during_evaluation_waits_for_execution_and_retains_its_outcome(
    tmp_path, outcome
):
    entered, release = Event(), Event()

    class HoldingEvaluator(Evaluator):
        def is_success(self, result):
            entered.set()
            assert release.wait(5)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    target_model = ScriptedModel(["One", "Two", "Three", "Four", "Presented"])
    attacker_model = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)
    user = FITDUser(
        LLMChat(target_model),
        LLMChat(attacker_model),
        Message("Custom request"),
        HoldingEvaluator(),
        artifacts_dir=tmp_path,
    )
    user.start()
    try:
        assert entered.wait(5)
        result = user.result()
        assert user.stop() is None
        assert user.status().state is UserState.STOPPING
        assert not user.wait(0)
    finally:
        release.set()
        assert user.wait(10)
    expected = UserState.FAILED if isinstance(outcome, Exception) else UserState.STOPPED
    assert user.status().state is expected
    assert user.result() is result
    assert len(target_model.requests) == 5
    assert len(attacker_model.requests) == 9
    _, metadata = read_and_assert_final_artifacts(user, tmp_path)
    if isinstance(outcome, Exception):
        assert metadata["evaluation"] is None
        assert "evaluation failed" in user.status().reason
    else:
        assert metadata["evaluation"] == {"result_id": result.id, "success": outcome}


def test_stop_during_artifact_finalization_is_responsive_and_updates_the_outcome(
    tmp_path, monkeypatch
):
    entered, release = Event(), Event()
    write_text = Path.write_text

    def hold_metadata(path, *args, **kwargs):
        if path.name == "metadata.json" and not entered.is_set():
            entered.set()
            assert release.wait(5), "test did not release artifact write"
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", hold_metadata)
    target = ScriptedModel(["One", "Two", "Three", "Four", "CONTROL"])
    attacker = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)
    user = FITDUser(
        LLMChat(target),
        LLMChat(attacker),
        Message("Custom request"),
        ContainsTextEvaluator("CONTROL"),
        artifacts_dir=tmp_path,
    )
    user.start()
    with ThreadPoolExecutor(max_workers=1) as control:
        try:
            assert entered.wait(5)
            assert user.status().state is UserState.RUNNING
            assert control.submit(user.stop).result(timeout=1) is None
            assert user.status().state is UserState.STOPPING
            assert not user.wait(0)
        finally:
            release.set()
            assert user.wait(10)
    assert user.status().state is UserState.STOPPED
    assert user.status().reason == "user_stop"
    assert user.status().evaluation.success is True
    assert len(target.requests) == 5
    assert len(attacker.requests) == 9
    events, metadata = read_and_assert_final_artifacts(user, tmp_path)
    finishes = [e["data"] for e in events if e["kind"] == "run_finished"]
    assert [e["revision"] for e in finishes] == [0, 1]
    assert metadata["finalization_revision"] == 1


def test_initial_request_exhaustion_completes_without_fabricating_a_result(tmp_path):
    attacker = LLMChat(ScriptedModel(["I cannot help with that"] * 3))
    target_model = ScriptedModel([])
    evaluator = RecordingEvaluator("CONTROL")
    user = FITDUser(
        LLMChat(target_model),
        attacker,
        Message("Custom request"),
        evaluator,
        artifacts_dir=tmp_path,
    )
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.COMPLETED, status.reason
    assert status.reason == "strategy_exhausted"
    assert status.progress.attacker_calls == 3
    assert target_model.requests == []
    assert evaluator.results == []
    assert user.result() is None
    assert status.evaluation is None
    run_dir = tmp_path / status.run_id
    assert not (run_dir / "result.txt").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["state"] == "COMPLETED"
    assert metadata["result"] is None
    assert metadata["evaluation"] is None
