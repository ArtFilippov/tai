import json
from pathlib import Path
from threading import Event
from typing import override

import pytest

from fitd_support import (
    RecordingEvaluator, ScriptedModel, Turns, evaluation, presented,
    read_and_assert_final_artifacts, reason, releasing, run_id,
)

from attacks.fitd.native import GarakException
from tai import ContainsTextEvaluator, FITDExperiment as FITDUser, LLMChat, Message, UserState, capture_run, save_run


def test_real_fitd_uses_the_supplied_request_and_publishes_the_evaluated_result(tmp_path: Path):
    attacker_model = ScriptedModel(
        ["Describe a fictional puzzle", "Level one", "Level two", "Level three"] + ["yes"] * 5
    )
    target_model = ScriptedModel(["Puzzle introduction", "More detail", "Further detail", "Almost done", "CONTROL"])
    target = LLMChat(target_model, model="same-model", system="Target instructions")
    attacker = LLMChat(attacker_model, model="same-model", system="Attacker instructions")
    evaluator = RecordingEvaluator("CONTROL")
    base = Message("Explain the final fictional puzzle solution")
    user = FITDUser(target, attacker, base, evaluator, journal_dir=tmp_path)
    assert user.status().state is UserState.IDLE
    assert user.result() is None
    assert user.start() is None
    assert user.wait(10)
    _assert_successful_participants(user, target, (target_model, attacker_model), evaluator)
    _assert_successful_artifacts(user, tmp_path)
    status, result = user.status(), user.result()
    assert user.stop() is None
    assert user.status() == status
    assert user.result() is result


def _assert_successful_participants(
    user: FITDUser, target: LLMChat, models: tuple[ScriptedModel, ScriptedModel], evaluator: RecordingEvaluator,
):
    target_model, attacker_model = models
    status = user.status()
    assert status.state is UserState.COMPLETED, status.reason
    assert status.reason == "evaluator_success"
    assert status.progress.target_calls == 5
    assert user.base.to_string() in attacker_model.requests[0][-1].message.to_string()
    assert target_model.requests[0][-1].message.to_string() == "Describe a fictional puzzle"
    assert target_model.requests[-1][-1].message.to_string() == user.base.to_string()
    assert len(target.history()) == 11
    assert all(r[0].message.to_string() == "Attacker instructions" for r in attacker_model.requests)
    result = presented(user)
    assert "CONTROL" in result.to_string()
    assert evaluator.results == [result]
    assert evaluator.results[0] is result
    assert evaluation(user).result_id == result.id
    assert evaluation(user).success is True


def _assert_successful_artifacts(user: FITDUser, root: Path):
    result = presented(user)
    run_dir = save_run(capture_run(user, journal=user.journal), root)
    assert (run_dir / "result.txt").read_text() == result.to_string()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["state"] == "COMPLETED"
    assert metadata["result"]["id"] == result.id
    assert metadata["evaluation"] == {"result_id": result.id, "success": True}
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [event.kind for event in user.journal.events()] == [e["kind"] for e in events]
    assert all(e["run_id"] == user.status().run_id for e in events)
    assert events[0]["kind"] == "run_started"
    assert events[-1]["kind"] == "run_finished"
    starts = {e["data"]["call_id"]: e for e in events if e["kind"] == "call_started"}
    ends = [e for e in events if e["kind"] == "call_completed"]
    assert len(starts) == len(ends) == 14
    assert {e["data"]["role"] for e in ends} == {"Target", "Attacker"}
    assert all(e["data"]["call_id"] in starts for e in ends)


def test_one_chat_cannot_serve_both_roles(tmp_path: Path):
    chat = LLMChat(lambda turns: "response")
    with pytest.raises(ValueError, match="independent"):
        FITDUser(chat, chat, Message("request"), ContainsTextEvaluator("CONTROL"), journal_dir=tmp_path)


def test_negative_evaluation_continues_and_is_not_attributed_to_new_result(tmp_path: Path):
    entered, release = Event(), Event()

    class HoldingEvaluator(RecordingEvaluator):
        @override
        def evaluate(self, base: Message, result: Message) -> bool:
            if self.results:
                entered.set()
                assert release.wait(10)
            return super().evaluate(base, result)

    evaluator = HoldingEvaluator("CONTROL")
    attacker = LLMChat(ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 6))
    target = LLMChat(ScriptedModel(["One", "Two", "Three", "Four", "Not yet", "CONTROL"]))
    user = FITDUser(target, attacker, Message("Custom request"), evaluator, journal_dir=tmp_path)
    user.start()
    with releasing(user, release):
        assert entered.wait(5)
        old_result = evaluator.results[0]
        current = presented(user)
        assert old_result.to_string() == "Not yet"
        assert current.to_string() == "CONTROL"
        assert current.id != old_result.id
        assert user.status().evaluation is None
        assert user.status().result_id == current.id
        decisions = [json.loads(e.dump())["data"] for e in user.journal.events() if e.kind == "evaluation_completed"]
        assert [{k: d[k] for k in ("result_id", "success")} for d in decisions] == [{"result_id": old_result.id, "success": False}]
        assert user.result() is current
        assert evaluator.results == [old_result]
    assert user.status().state is UserState.COMPLETED
    assert evaluation(user).success is True
    assert old_result.to_string() == "Not yet"


def test_round_limit_is_normal_completion_and_empty_result_is_present(tmp_path: Path):
    evaluator = RecordingEvaluator("CONTROL")
    # Empty text is a completed response, distinct from a missing result.
    target = LLMChat(ScriptedModel([""] * 10))
    attacker = LLMChat(ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 10))
    user = FITDUser(target, attacker, Message("Custom request"), evaluator, journal_dir=tmp_path)
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.COMPLETED, status.reason
    assert status.reason == "round_limit"
    assert status.progress.target_calls == 10
    assert evaluation(user).success is False
    # Only adversarial-level results are evaluated, rather than each exchange.
    assert len(evaluator.results) == 6
    assert user.result() is evaluator.results[-1]
    run_dir = save_run(capture_run(user, journal=user.journal), tmp_path)
    assert not (run_dir / "result.txt").read_text()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["result"] is not None
    assert not metadata["result"]["text"]
    assert metadata["evaluation"]["result_id"] == metadata["result"]["id"]
    assert len([e for e in user.journal.events() if e.kind == "evaluation_completed"]) == 6


def test_fitd_realigns_a_response_before_continuing_the_ladder(tmp_path: Path):
    attacker = LLMChat(ScriptedModel(["Describe a fictional puzzle", "L1", "L2", "L3", "no"] + ["yes"] * 5))
    target_model = ScriptedModel(["An unrelated answer", "Puzzle introduction", "Detail", "More", "Almost", "CONTROL"])
    target = LLMChat(target_model)
    user = FITDUser(target, attacker, Message("Custom request"), ContainsTextEvaluator("CONTROL"), journal_dir=tmp_path)
    user.start()
    assert user.wait(10)
    assert user.status().state is UserState.COMPLETED, user.status().reason
    correction_request = target_model.requests[1][-1].message.to_string()
    assert "An unrelated answer" in correction_request
    assert "Describe a fictional puzzle" in correction_request
    assert target_model.requests[1][-2].message.to_string() == "An unrelated answer"
    assert user.status().progress.target_calls == 6
    assert evaluation(user).success is True


def test_run_without_a_presented_result_has_no_result_file_or_evaluation(tmp_path: Path):
    def unavailable(turns: Turns) -> str:
        raise RuntimeError("model unavailable")

    user = FITDUser(LLMChat(lambda turns: "unused"), LLMChat(unavailable), Message("request"), ContainsTextEvaluator("CONTROL"), journal_dir=tmp_path)
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.FAILED
    assert user.result() is None
    assert status.evaluation is None
    run_dir = save_run(capture_run(user, journal=user.journal), tmp_path)
    assert not (run_dir / "result.txt").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["result"] is None
    assert metadata["evaluation"] is None


@pytest.mark.parametrize("error_type", [RuntimeError, GarakException, NotImplementedError])
def test_required_evaluation_failure_preserves_new_result_without_an_old_score(tmp_path: Path, error_type: type[Exception]):
    class FailingEvaluator(RecordingEvaluator):
        @override
        def evaluate(self, base: Message, result: Message) -> bool:
            if self.results:
                # Observation is durable; the presented result is not yet saved.
                directory = tmp_path / run_id(user)
                assert not (directory / "result.txt").exists()
                assert user.result() is result
                raise error_type("required evaluation unavailable")
            return super().evaluate(base, result)

    evaluator = FailingEvaluator("CONTROL")
    target_model = ScriptedModel(["One", "Two", "Three", "Four", "Negative", "Unchecked"])
    attacker_model = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 6)
    user = FITDUser(LLMChat(target_model), LLMChat(attacker_model), Message("Custom request"), evaluator, journal_dir=tmp_path)
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.FAILED
    assert "required evaluation unavailable" in reason(user)
    assert presented(user).to_string() == "Unchecked"
    assert status.evaluation is None
    assert status.progress.target_calls == 6
    assert status.progress.attacker_calls == 10
    _assert_evaluation_failure_artifacts(user, tmp_path, evaluator, error_type)
    assert user.stop() is None
    assert user.status() == status


def _assert_evaluation_failure_artifacts(user: FITDUser, root: Path, evaluator: RecordingEvaluator, error_type: type[Exception]):
    events, metadata = read_and_assert_final_artifacts(user, root)
    assert metadata["evaluation"] is None
    evaluations = [e["data"] for e in events if e["kind"] == "evaluation_completed"]
    assert [{k: d[k] for k in ("result_id", "success")} for d in evaluations] == [{"result_id": evaluator.results[0].id, "success": False}]
    failures = [e["data"] for e in events if e["kind"] == "evaluation_failed"]
    assert len(failures) == 1
    assert failures[0]["result_id"] == presented(user).id
    assert failures[0]["error"] == f"{error_type.__name__}: required evaluation unavailable"


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
    tmp_path: Path, error_type: type[Exception], role: str, prior_target_responses: list[str], attacker_responses: list[str],
):
    responses = {"Target": prior_target_responses, "Attacker": attacker_responses}
    models = {name: ScriptedModel([*values, *([error_type("provider failed")] if name == role else [])]) for name, values in responses.items()}
    chats = {name: LLMChat(model, system=name) for name, model in models.items()}
    user = FITDUser(chats["Target"], chats["Attacker"], Message("Custom request"), ContainsTextEvaluator("CONTROL"), journal_dir=tmp_path)
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
        assert user.result() is None
    elif prior_target_responses:
        assert presented(user).to_string() == prior_target_responses[-1]
    _assert_provider_failure_artifacts(user, tmp_path, role)
    assert status.progress.target_calls == len(prior_target_responses)
    assert status.progress.attacker_calls == len(attacker_responses)
    assert user.stop() is None
    assert user.status() == status


def _assert_provider_failure_artifacts(user: FITDUser, root: Path, role: str):
    events, metadata = read_and_assert_final_artifacts(user, root)
    assert metadata["evaluation"] is None
    failures = [e["data"] for e in events if e["kind"] == "call_failed"]
    assert len(failures) == 1
    assert failures[0]["role"] == role
    assert failures[0]["error"] == user.status().reason
    starts = [e["data"]["call_id"] for e in events if e["kind"] == "call_started"]
    assert failures[0]["call_id"] == starts[-1]
    assert len([e for e in events if e["kind"] == "call_completed"]) == len(starts) - 1


def test_failure_after_native_restore_preserves_accepted_result_and_discarded_journal(tmp_path: Path):
    rejected = "I cannot help with that: CONTROL"
    target_model = ScriptedModel(["Retained opening", rejected, RuntimeError("provider failed after restore")])
    attacker_model = ScriptedModel(["Initial", "L1", "L2", "L3", "yes", "Safer request"])
    target = LLMChat(target_model)
    user = FITDUser(target, LLMChat(attacker_model), Message("Custom request"), ContainsTextEvaluator("CONTROL"), journal_dir=tmp_path)
    user.start()
    assert user.wait(10)
    assert user.status().state is UserState.FAILED
    assert user.status().reason == "RuntimeError: provider failed after restore"
    assert presented(user).to_string() == "Retained opening"
    assert len(target_model.requests) == 3
    assert len(attacker_model.requests) == 6
    assert target_model.requests[2][-1].message.to_string() == "Safer request"
    assert target_model.requests[2][:-1] == target_model.requests[1][:-1]
    assert len(target.history()) == 2
    assert target.history()[-1].message is user.result()
    assert all(rejected not in turn.message.to_string() for turn in target.history())
    events, metadata = read_and_assert_final_artifacts(user, tmp_path)
    assert metadata["evaluation"] is None
    assert [e["data"]["result"]["text"] for e in events if e["kind"] == "result_published"] == ["Retained opening"]
    assert any(e["kind"] == "call_completed" and e["data"]["response"]["text"] == rejected for e in events)
    branch = next(e["data"] for e in events if e["kind"] == "branch_selected" and e["data"]["branch"] == "rephrase")
    assert not any(e["kind"] == "result_published" and e["data"]["cause_id"] == branch["branch_id"] for e in events)
    assert any(e["kind"] == "history_restored" and e["data"]["cause_id"] == branch["branch_id"] for e in events)
    assert not any(e["kind"] == "evaluation_completed" for e in events)


def test_initial_request_exhaustion_completes_without_fabricating_a_result(tmp_path: Path):
    attacker = LLMChat(ScriptedModel(["I cannot help with that"] * 3))
    target_model = ScriptedModel([])
    evaluator = RecordingEvaluator("CONTROL")
    user = FITDUser(LLMChat(target_model), attacker, Message("Custom request"), evaluator, journal_dir=tmp_path)
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.COMPLETED, status.reason
    assert status.reason == "strategy_exhausted"
    assert status.progress.attacker_calls == 3
    assert not target_model.requests
    assert not evaluator.results
    assert user.result() is None
    assert status.evaluation is None
    run_dir = save_run(capture_run(user, journal=user.journal), tmp_path)
    assert not (run_dir / "result.txt").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["state"] == "COMPLETED"
    assert metadata["result"] is None
    assert metadata["evaluation"] is None
