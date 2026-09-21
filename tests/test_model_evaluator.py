"""Explicit-pair classification and strict parsing, with controlled responses."""

import json

import pytest
from test_fitd import ScriptedModel, read_and_assert_final_artifacts

from tai import (
    EvaluatorSetup,
    LLMChat,
    Message,
    ModelEvaluationError,
    ModelEvaluator,
    ModelVerdictError,
    QwenContextError,
    UserState,
)
from tai import (
    FITDExperiment as FITDUser,
)
from tai.evaluators import MODEL_RUBRIC, MODEL_RUBRIC_VERSION


def classification_data(turns):
    rubric, data = turns[-1].message.to_string().split("\n\nINPUT_JSON:\n", 1)
    assert rubric == MODEL_RUBRIC
    return json.loads(data)


@pytest.mark.parametrize(
    ("response", "success"),
    [(' {"success": true}\n', True), ('\n{"success": false} ', False)],
)
def test_model_returns_only_the_structured_boolean(response, success):
    model = ScriptedModel([response])
    evaluator = ModelEvaluator(LLMChat(model))
    assert (
        evaluator.evaluate(Message("Explicit base"), Message("Presented result"))
        is success
    )
    assert len(model.requests) == 1


@pytest.mark.parametrize(
    "response",
    [
        "",
        " \n\t",
        "true",
        "false",
        "yes, success",
        "null",
        "[]",
        '{"success": true',
        '```json\n{"success": true}\n```',
        'Verdict: {"success": false}',
        '{"success": true} trailing text',
        '{"success": false}{"success": true}',
        "{}",
        '{"Success": true}',
        '{"success": "false"}',
        '{"success": 0}',
        '{"success": 1}',
        '{"success": null}',
        '{"success": []}',
        '{"success": {}}',
        '{"success": NaN}',
        '{"success": Infinity}',
        '{"success": True}',
        '{"success": true, "success": false}',
        '{"success": false, "reason": "refusal"}',
    ],
)
def test_unknown_or_ambiguous_verdict_is_an_error_never_false(response):
    chat = LLMChat(lambda turns: response)
    evaluator = ModelEvaluator(chat)
    with pytest.raises(ModelVerdictError, match="boolean"):
        evaluator.evaluate(Message("Base"), Message("Result"))
    # Malformed verdicts are still completed model exchanges, available intact.
    assert chat.history()[-1].message.to_string() == response


def test_full_current_text_and_explicit_base_are_data_and_assessments_are_isolated():
    model = ScriptedModel(
        ["Old conversation", '{"success": false}', '{"success": true}']
    )
    chat = LLMChat(model, system="Classifier system")
    chat.send(Message("Unrelated prior base"))
    old_point = chat.checkpoint()
    base = Message("Original base").transform(str.upper)
    # The base request is deliberately not the origin or transformation root.
    full = 'Prefix\n"result": "injected"\n' + "Unicode π\n" * 20000 + "END\n"
    result = Message("Discarded base").spawn("Old result").transform(lambda _: full)
    evaluator = ModelEvaluator(chat)
    assert evaluator.evaluate(base, result) is False
    first_id = chat.conversation_id
    first_point = chat.checkpoint()
    assert classification_data(model.requests[1]) == {
        "base_request": base.to_string(),
        "result": full,
    }
    other_base = Message("Independent next base")
    assert evaluator.evaluate(other_base, Message("Next result")) is True
    assert classification_data(model.requests[2]) == {
        "base_request": "Independent next base",
        "result": "Next result",
    }
    assert chat.conversation_id != first_id
    assert all(
        [turn.role for turn in turns] == ["system", "user"] for turns in model.requests
    )
    assert all(
        turns[0].message.to_string() == "Classifier system" for turns in model.requests
    )
    for point in (old_point, first_point):
        with pytest.raises(ValueError, match="stale"):
            chat.restore(point)
    setup = EvaluatorSetup.model(chat)
    settings = setup.settings()
    assert "base" not in settings
    assert settings["rubric_version"] == MODEL_RUBRIC_VERSION == "alert-safety-v1"
    assert settings["rubric"] == MODEL_RUBRIC
    settings["chat"]["initial_history"][0]["message"]["text"] = "Mutation of snapshot"
    assert (
        setup.settings()["chat"]["initial_history"][0]["message"]["text"]
        == "Classifier system"
    )
    assert len(model.requests) == 3  # Observing configuration never sends.


@pytest.mark.parametrize(
    "error", [RuntimeError("model unavailable"), QwenContextError("context overflow")]
)
def test_model_failure_is_explicit_with_original_cause_and_no_completed_exchange(error):
    chat = LLMChat(ScriptedModel([error]))
    evaluator = ModelEvaluator(chat)
    with pytest.raises(ModelEvaluationError, match=str(error)) as raised:
        evaluator.evaluate(Message("base"), Message("result"))
    assert raised.value.__cause__ is error
    assert not chat.history()


def test_model_evaluator_requires_pinned_rubric_and_explicit_message_base():
    chat = LLMChat(lambda turns: "unused")
    with pytest.raises(TypeError, match="explicit base and result Messages"):
        ModelEvaluator(chat).evaluate("base", Message("result"))
    with pytest.raises(ValueError, match="rubric_version"):
        ModelEvaluator(chat, "latest")


def test_model_evaluator_uses_the_rubric_supplied_at_construction():
    requests = []
    evaluator = ModelEvaluator(
        LLMChat(lambda turns: requests.append(turns) or '{"success": false}'),
        rubric="Classify the explicit pair; return a JSON boolean success field.",
    )
    assert evaluator.evaluate(Message("Base"), Message("Result")) is False
    rubric, payload = requests[0][-1].message.to_string().split("\n\nINPUT_JSON:\n")
    assert rubric == evaluator.rubric
    assert json.loads(payload) == {"base_request": "Base", "result": "Result"}


@pytest.mark.parametrize("shared", ["target", "attacker"])
def test_fitd_rejects_shared_third_role_before_any_execution(tmp_path, shared):
    target, attacker = [LLMChat(lambda turns: "unused") for _ in range(2)]
    base = Message("base")
    evaluator = EvaluatorSetup.model({"target": target, "attacker": attacker}[shared])
    with pytest.raises(ValueError, match="independent"):
        FITDUser(target, attacker, base, evaluator, journal_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def make_user(tmp_path, judge, *, rounds=6, base=None):
    base = base if base is not None else Message("The explicitly supplied goal")
    target_model = ScriptedModel(
        ["One", "Two", "Three", "Four", "First result", "Final π\n" * 3000]
    )
    attacker_model = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 6)
    target = LLMChat(target_model, system="Target system")
    attacker = LLMChat(attacker_model, system="Attacker system")
    evaluator = EvaluatorSetup.model(judge)
    user = FITDUser(
        target,
        attacker,
        base,
        evaluator,
        journal_dir=tmp_path,
        max_calls_per_conv=rounds,
    )
    return user, target_model, attacker_model, evaluator


@pytest.mark.parametrize("verdicts", [[True], [False, True], [False, False]])
def test_real_pinned_fitd_uses_boolean_decisions_and_saves_message_linked_exchanges(
    tmp_path, verdicts
):
    model = ScriptedModel([json.dumps({"success": value}) for value in verdicts])
    judge = LLMChat(model, system="Evaluator system")
    user, target, attacker, evaluator = make_user(tmp_path, judge)

    user.start()
    assert user.wait(10)
    status, result = user.status(), user.result()
    assert status.state is UserState.COMPLETED, status.reason
    assert status.reason == ("evaluator_success" if verdicts[-1] else "round_limit")
    assert status.progress.target_calls == len(verdicts) + 4
    assert status.progress.attacker_calls == len(verdicts) + 8
    assert status.progress.evaluator_calls == len(verdicts)
    assert status.evaluation.result_id == result.id
    assert status.evaluation.success is verdicts[-1]
    events, metadata = read_and_assert_final_artifacts(user, tmp_path)
    assert (
        metadata["settings"]["garak_revision"]
        == "3f50ea5ff9cd7050099940647c15c39b07a93392"
    )
    assert metadata["settings"]["evaluator"]["rubric"] == MODEL_RUBRIC
    assert metadata["settings"]["base"]["id"] == user._base.id
    starts = [
        e["data"]
        for e in events
        if e["kind"] == "call_started" and e["data"]["role"] == "Evaluator"
    ]
    ends = [
        e["data"]
        for e in events
        if e["kind"] == "call_completed" and e["data"]["role"] == "Evaluator"
    ]
    evaluations = [e["data"] for e in events if e["kind"] == "evaluation_completed"]
    assessments = [e["data"] for e in events if e["kind"] == "evaluation_started"]
    assert len({e["evaluation_id"] for e in assessments}) == len(verdicts)
    assert [e["success"] for e in evaluations] == verdicts
    published = {
        e["data"]["result"]["id"]: e["data"]["result"]
        for e in events
        if e["kind"] == "result_published"
    }
    assert len({s["conversation_id"] for s in starts}) == len(verdicts)
    assert all(len(turns) == 2 for turns in model.requests)
    assert not {s["conversation_id"] for s in starts} & {
        e["data"]["conversation_id"]
        for e in events
        if e["kind"] == "call_started" and e["data"]["role"] != "Evaluator"
    }
    for assessment, start, end, evaluation, turns in zip(
        assessments, starts, ends, evaluations, model.requests, strict=True
    ):
        assert (
            assessment["base_id"]
            == start["base_id"]
            == evaluation["base_id"]
            == user._base.id
        )
        assert assessment["base"] == json.loads(user._base.dump())
        assert assessment["result"] == published[start["result_id"]]
        assert (
            assessment["evaluation_id"]
            == start["evaluation_id"]
            == end["evaluation_id"]
            == evaluation["evaluation_id"]
        )
        assert start["call_id"] == end["call_id"]
        assert start["result_id"] == end["result_id"] == evaluation["result_id"]
        assert start["input"]["id"] == end["response"]["origin"]["id"]
        assert classification_data(turns) == {
            "base_request": user._base.to_string(),
            "result": published[start["result_id"]]["text"],
        }
        assert json.loads(end["response"]["text"])["success"] is evaluation["success"]
    # Reading the public status, result, artifacts and settings does not re-evaluate.
    assert user.result() is result and user.status() == status
    evaluator.settings()
    assert len(model.requests) == len(verdicts)
    assert len(target.requests) == status.progress.target_calls
    assert len(attacker.requests) == status.progress.attacker_calls


@pytest.mark.parametrize(
    "failure",
    [
        '{"success": "false"}',
        RuntimeError("judge unavailable"),
        QwenContextError("full input overflow"),
    ],
)
def test_real_fitd_assessment_error_is_not_a_negative_score_and_keeps_raw_evidence(
    tmp_path, failure
):
    model = ScriptedModel(['{"success": false}', failure])
    user, target, attacker, _ = make_user(tmp_path, LLMChat(model))
    user.start()
    assert user.wait(10)
    assert user.status().state is UserState.FAILED
    assert user.status().evaluation is None
    assert user.result().to_string() == "Final π\n" * 3000
    assert len(target.requests) == 6 and len(attacker.requests) == 10
    events, metadata = read_and_assert_final_artifacts(user, tmp_path)
    assert metadata["evaluation"] is None
    failures = [e["data"] for e in events if e["kind"] == "evaluation_failed"]
    assert len(failures) == 1 and failures[0]["result_id"] == user.result().id
    assert failures[0]["error"] == user.status().reason
    if isinstance(failure, str):
        assert "ModelVerdictError" in user.status().reason
        raw = [e["data"] for e in events if e["kind"] == "call_completed"][-1]
        assert raw["response"]["text"] == failure
    else:
        assert "ModelEvaluationError" in user.status().reason
        raw = [e["data"] for e in events if e["kind"] == "call_failed"][-1]
        assert raw["error"] == f"{type(failure).__name__}: {failure}"
    assert raw["role"] == "Evaluator" and raw["result_id"] == user.result().id


def test_discarded_real_fitd_branch_cannot_supply_classification_evidence(tmp_path):
    rejected = "I cannot help with that: discarded success claim"
    target = LLMChat(
        ScriptedModel(
            ["Opening", rejected, "Gentle", "Connected", "Draft", "Presented"]
        )
    )
    attacker = LLMChat(
        ScriptedModel(["Initial", "L1", "L2", "L3", "yes", "Rephrased"] + ["yes"] * 5)
    )
    model = ScriptedModel(['{"success": false}'])
    base = Message("Supplied goal")
    user = FITDUser(
        target,
        attacker,
        base,
        EvaluatorSetup.model(LLMChat(model)),
        journal_dir=tmp_path,
        max_calls_per_conv=6,
    )
    user.start()
    assert user.wait(10)
    assert user.status().state is UserState.COMPLETED, user.status().reason
    assert user.status().evaluation.success is False
    assert classification_data(model.requests[0]) == {
        "base_request": "Supplied goal",
        "result": "Presented",
    }
    assert len(model.requests) == 1
    events, _ = read_and_assert_final_artifacts(user, tmp_path)
    assert any(e["kind"] == "history_restored" for e in events)
    assert any(
        e["kind"] == "call_completed" and e["data"]["response"]["text"] == rejected
        for e in events
    )
