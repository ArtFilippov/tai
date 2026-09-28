"""Pinned GOAT prompts, decisions and independent role conversations."""

import json
from pathlib import Path

import pytest
from evaluation_fakes import HeldResponse, ScriptedModel
from fitd_support import releasing
from test_adaptive_attacks import PairEvaluator

from tai import ContainsTextEvaluator, GOATExperiment, GOATUser, LLMChat, Message, UserState, capture_run, save_run


def otsr(response: str) -> str:
    return f"Observation: previous turn\nThought: adapt\nStrategy: revise\nResponse: {response}"


@pytest.mark.parametrize("judge,success_at,reason", [
    ("safe", None, "round_limit"), ("unsafe", None, "judge_stop"), ("safe", 2, "evaluator_success"),
])
def test_two_turns_use_native_protocol_and_evaluate_only_presented_results(
    tmp_path: Path, judge: str, success_at: int | None, reason: str,
) -> None:
    base = Message("Prior").transform(lambda _: "Explain the vault")
    attacker_model = ScriptedModel([otsr("Try first"), "safe", otsr("Try after feedback"), judge])
    target_model = ScriptedModel(["First reply", "Complete reply π\n"])
    evaluator = PairEvaluator(success_at)
    attacker, target = LLMChat(attacker_model), LLMChat(target_model)
    user = GOATExperiment(target, attacker, base, evaluator, journal_dir=tmp_path, max_calls_per_conv=2)
    user.start()
    assert user.wait(10) and user.status().state is UserState.COMPLETED, user.status().reason
    assert user.status().reason == reason
    assert [turns[-1].message.text for turns in target_model.requests] == ["Try first", "Try after feedback"]
    assert len(target_model.requests[1]) == 3
    assert "First reply" in attacker_model.requests[2][-1].message.text
    assert len(attacker_model.requests[2]) == 3  # Judge's one-shot exchange was restored.
    assert [pair[0] for pair in evaluator.pairs] == [base, base]
    assert evaluator.pairs[-1][1] is user.result()
    directory = save_run(capture_run(user, journal=user.journal), tmp_path)
    assert (directory / "result.txt").read_text() == "Complete reply π\n"
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    judges = [event["data"] for event in events if event["kind"] == "local_decision" and event["data"]["decision"] == "goat_judge"]
    assessments = [event["data"] for event in events if event["kind"] == "evaluation_completed"]
    assert [item["value"] for item in judges] == [False, judge == "unsafe"]
    assert [item["success"] for item in assessments] == [False, success_at == 2]
    result = user.result()
    assert result is not None
    assert judges[-1]["response_id"] == assessments[-1]["result_id"] == result.id


def test_early_judge_stop_does_not_claim_evaluator_success() -> None:
    user = GOATExperiment(LLMChat(ScriptedModel(["Complete"])), LLMChat(ScriptedModel([otsr("Ask"), "unsafe"])),
                          Message("Base"), ContainsTextEvaluator("absent"), max_calls_per_conv=3)
    user.start()
    assert user.wait(10) and user.status().reason == "judge_stop"
    result, evaluation = user.result(), user.status().evaluation
    assert result is not None and result.text == "Complete"
    assert evaluation is not None and not evaluation.success


def test_stop_during_target_retains_completed_material_without_presenting_it() -> None:
    held = HeldResponse("Target answer")
    user = GOATExperiment(LLMChat.cancellable(ScriptedModel([held])), LLMChat(ScriptedModel([otsr("Ask")])),
                          Message("Base"), ContainsTextEvaluator("Target"))
    user.start()
    with releasing(user, held.release):
        assert held.entered.wait(5)
        user.stop()
    assert user.status().state is UserState.STOPPED and user.result() is None
    events = [json.loads(event.dump()) for event in user.journal.events()]
    assert any(event["kind"] == "call_completed" and event["data"]["role"] == "Target" for event in events)


def test_target_failure_preserves_error_and_releases_participants() -> None:
    target = ScriptedModel([RuntimeError("target unavailable")])
    user = GOATExperiment(LLMChat(target), LLMChat(ScriptedModel([otsr("Ask")])),
                          Message("Base"), ContainsTextEvaluator("yes"))
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.FAILED and status.reason is not None and "target unavailable" in status.reason
    assert user.result() is None and user.journal.finished
    assert any(event.kind == "resources_released" for event in user.journal.events())


def test_invalid_options_and_shared_chat_fail_before_start() -> None:
    chat = LLMChat(str)
    with pytest.raises(ValueError, match="independent"):
        GOATUser(chat, chat, Message("Base"), ContainsTextEvaluator("yes"))
    with pytest.raises(ValueError, match="max_calls_per_conv"):
        GOATUser(chat, LLMChat(str), Message("Base"), ContainsTextEvaluator("yes"), max_calls_per_conv=0)


def test_repeated_starts_get_independent_journals_and_conversations() -> None:
    attacker_model = ScriptedModel([otsr("First"), "safe", otsr("Second"), "safe"])
    target_model = ScriptedModel(["First result", "Second result"])
    user = GOATExperiment(LLMChat(target_model), LLMChat(attacker_model), Message("Base"),
                          ContainsTextEvaluator("never"), max_calls_per_conv=1)
    user.start()
    assert user.wait(10)
    first = user.result()
    assert first is not None and first.text == "First result"
    first_journal = user.journal
    user.start()
    assert user.wait(10)
    second = user.result()
    assert second is not None and second.text == "Second result"
    assert len(user.journals) == 2 and user.journal is not first_journal
    assert target_model.requests[0][-1].message.text == "First"
    assert target_model.requests[1][-1].message.text == "Second"
    assert len(target_model.requests[1]) == 1
