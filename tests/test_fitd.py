import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import Event

import pytest

from tai import ContainsTextEvaluator, Evaluator, FITDUser, LLMChat, Message, UserState


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def __call__(self, turns):
        self.requests.append(turns)
        return next(self.responses)


class RecordingEvaluator(Evaluator):
    def __init__(self, criterion):
        self.evaluator = ContainsTextEvaluator(criterion)
        self.results = []

    def is_success(self, result):
        self.results.append(result)
        return self.evaluator.is_success(result)


def test_real_fitd_uses_the_supplied_request_and_publishes_the_evaluated_result(
    tmp_path,
):
    # Initial safe request, three ladder levels, then alignment decisions.
    attacker_model = ScriptedModel(
        ["Describe a fictional puzzle", "Level one", "Level two", "Level three"]
        + ["yes"] * 4
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
    assert len(starts) == len(ends) == 13
    assert {e["data"]["role"] for e in ends} == {"Target", "Attacker"}
    assert all(e["data"]["call_id"] in starts for e in ends)


@pytest.mark.parametrize("held_role", ["Target", "Attacker"])
def test_control_methods_remain_responsive_during_model_calls(tmp_path, held_role):
    entered, release = Event(), Event()
    attacker_model = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 4)
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
                assert result.to_string() == "First"
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
                assert target.history()[-1].message is result
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
    attacker = LLMChat(ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5))
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
            ["Describe a fictional puzzle", "L1", "L2", "L3", "no"] + ["yes"] * 4
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
