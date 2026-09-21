"""One-operation extensions through the actual FITD/readiness/series paths."""

import json
from pathlib import Path
from threading import Event

import pytest
from test_fitd import ScriptedModel

from tai import (
    EvaluatorSetup,
    FITDExperiment,
    FITDUser,
    LLMChat,
    Message,
    UserState,
)
from app.configuration import RoleProfile
from app.preflight import run_preflight
from app.series import run_alert_series


def participants(repeats=1):
    return (
        LLMChat(ScriptedModel(["One", "Two", "Three", "Four", "Presented"] * repeats)),
        LLMChat(ScriptedModel((["Initial", "L1", "L2", "L3"] + ["yes"] * 5) * repeats)),
    )


@pytest.mark.parametrize("user_type", [FITDUser, FITDExperiment])
def test_one_operation_receives_exact_messages_across_independent_runs(user_type):
    pairs = []

    class CustomEvaluator:
        __slots__ = ()

        def evaluate(self, base, result):
            pairs.append((base, result))
            return base.to_string() == "Second base"

    evaluator = CustomEvaluator()
    target, attacker = participants(3)
    bases = [Message("First base"), Message("Second base")]
    users = []
    for base in bases:
        user = user_type(target, attacker, base, evaluator, max_calls_per_conv=5)
        user.start()
        assert user.wait(10)
        assert user.status().state is UserState.COMPLETED, user.status().reason
        assert pairs[-1][0] is base and pairs[-1][1] is user.result()
        assert user.status().evaluation.success is (base is bases[1])
        users.append(user)
    old = users[-1].snapshot()
    users[-1].start()
    assert users[-1].wait(10)
    assert users[-1].result() is not old.result
    assert users[-1].status().evaluation.result_id == users[-1].result().id
    assert len(pairs) == 3
    if user_type is FITDExperiment:
        for user in users:
            for journal in user.journals:
                settings = journal.settings()
                settings["base"]["text"] = "Edited diagnostic"
                starts = [
                    e.data for e in journal.events() if e.kind == "evaluation_started"
                ]
                ends = [
                    e.data for e in journal.events() if e.kind == "evaluation_completed"
                ]
                assert len(starts) == len(ends) == 1
                assert starts[0]["evaluation_id"] == ends[0]["evaluation_id"]
                assert starts[0]["base_id"] == ends[0]["base_id"] == user._base.id
                assert starts[0]["result"]["id"] == ends[0]["result_id"]
                assert journal.settings()["base"]["text"] == user._base.to_string()
        assert len(pairs) == 3


@pytest.mark.parametrize(
    "verdict", [True, False, None, 0, 1, "false", [], RuntimeError("unavailable")]
)
def test_preflight_and_fitd_enforce_the_same_custom_evaluator_contract(verdict, capsys):
    pairs = []

    class CustomEvaluator:
        def evaluate(self, base, result):
            pairs.append((base, result))
            if isinstance(verdict, Exception):
                raise verdict
            return verdict

    evaluator = CustomEvaluator()
    profile = RoleProfile(
        LLMChat(lambda turns: "READY"),
        LLMChat(lambda turns: "READY"),
        EvaluatorSetup(evaluator),
        5,
    )
    valid = type(verdict) is bool
    assert run_preflight(profile) == (0 if valid else 1)
    report = json.loads(capsys.readouterr().out)
    assessment = report["roles"][2]
    assert assessment["ok"] is valid
    assert pairs[0][0].to_string() == "Name the capital of France."
    if valid:
        assert assessment["success"] is verdict
    else:
        assert "success" not in assessment
        assert (
            "unavailable" if isinstance(verdict, Exception) else "must return bool"
        ) in assessment["error"]
    user = FITDExperiment(
        *participants(), Message("Actual base"), evaluator, max_calls_per_conv=5
    )
    user.start()
    assert user.wait(10)
    assert user.status().state is (UserState.COMPLETED if valid else UserState.FAILED)
    assert len(pairs) == 2
    assert pairs[1][0] is user._base and pairs[1][1] is user.result()
    if valid:
        assert user.status().evaluation.success is verdict
    else:
        assert user.status().evaluation is None
        failure = next(
            e.data for e in user.journal.events() if e.kind == "evaluation_failed"
        )
        assert failure["base_id"] == user._base.id
        assert failure["result_id"] == user.result().id


def test_stop_waits_for_custom_assessment_and_diagnostic_mutation_cannot_change_input():
    entered, release = Event(), Event()
    pairs = []
    configuration = {"expected": "Presented"}

    class CustomEvaluator:
        def evaluate(self, base, result):
            pairs.append((base, result))
            entered.set()
            assert release.wait(10)
            assert base.to_string() == "Base"
            assert result.to_string() == configuration["expected"] == "Presented"
            return False

    setup = EvaluatorSetup(CustomEvaluator(), configuration=lambda: configuration)
    user = FITDExperiment(*participants(), Message("Base"), setup)
    user.start()
    try:
        assert entered.wait(10)
        result = user.result()
        event = next(e for e in user.journal.events() if e.kind == "evaluation_started")
        copy = event.data
        copy["base"]["text"] = "Edited base"
        copy["result"]["text"] = "Edited result"
        setup.settings()["expected"] = "Edited criterion"
        user.journal.settings()["evaluator"]["expected"] = "Edited criterion"
        user.stop()
        assert user.status().state is UserState.STOPPING
        assert not user.wait(0)
    finally:
        release.set()
        assert user.wait(10)
    assert user.status().state is UserState.STOPPED
    assert user.status().evaluation.result_id == result.id
    assert user.status().evaluation.success is False
    assert pairs == [(user._base, result)]
    events = user.journal.events()
    stop = next(e.sequence for e in events if e.kind == "stop_requested")
    assert not any(
        e.kind in {"call_started", "evaluation_started"} and e.sequence > stop
        for e in events
    )


def test_series_uses_the_assembled_one_operation_evaluator(tmp_path, monkeypatch):
    pairs = []

    class CustomEvaluator:
        def evaluate(self, base, result):
            pairs.append((base, result))
            return True

    profile = RoleProfile(*participants(32), EvaluatorSetup(CustomEvaluator()), 20)
    monkeypatch.setattr("app.series.load_profile", lambda *args, **kwargs: profile)
    series = run_alert_series(
        Path("examples/alert-manifest.json"),
        Path("examples/fitd-alert-qwen.toml"),
        tmp_path,
        taxonomy_path=Path("ALERT_taxonomy.md"),
    )
    assert series.exit_code == 0
    index = json.loads((series.directory / "index.json").read_text())
    assert len(pairs) == len(index["entries"]) == 32
    assert len({base.id for base, _ in pairs}) == 32
    assert len({result.id for _, result in pairs}) == 32
    for (base, result), entry in zip(pairs, index["entries"], strict=True):
        assert entry["base_id"] == base.id
        assert entry["evaluation"] == {"result_id": result.id, "success": True}


@pytest.mark.parametrize("stop", [False, True])
def test_custom_chat_evaluator_is_observed_and_owned_without_class_specific_io(stop):
    entered, release = Event(), Event()
    requests = []

    def generate(turns):
        requests.append(turns)
        if len(requests) == 1:
            entered.set()
            assert release.wait(10)
        return "allow"

    class TwoExchangeEvaluator:
        def __init__(self, chat):
            self.chat = chat

        def evaluate(self, base, result):
            self.chat.reset()
            self.chat.send(base)
            return self.chat.send(result).to_string() == "allow"

    judge = LLMChat(generate, system="Independent classifier")
    setup = EvaluatorSetup.with_chat(judge, TwoExchangeEvaluator)
    user = FITDExperiment(*participants(), Message("Actual base"), setup)
    contender = FITDUser(*participants(), Message("Other"), setup)
    user.start()
    try:
        assert entered.wait(10)
        with pytest.raises(RuntimeError, match="active execution"):
            contender.start()
        if stop:
            user.stop()
    finally:
        release.set()
        assert user.wait(10)
    assert user.status().state is (UserState.STOPPED if stop else UserState.COMPLETED)
    assert user.status().evaluation.success is True
    assert requests[0][-1].message is user._base
    assert requests[1][-1].message is user.result()
    starts = [e.data for e in user.journal.events() if e.kind == "evaluation_started"]
    calls = [
        e.data
        for e in user.journal.events()
        if e.kind == "call_completed" and e.data["role"] == "Evaluator"
    ]
    assert len(starts) == 1 and len(calls) == 2
    assert all(c["evaluation_id"] == starts[0]["evaluation_id"] for c in calls)
    assert all(c["result_id"] == user.result().id for c in calls)
    assert user.status().progress.evaluator_calls == 1
    assert judge._owner is None


def test_failed_preflight_does_not_leave_observation_bound_to_shared_evaluator(capsys):
    judge = LLMChat(ScriptedModel(["invalid JSON", '{"success": false}']))
    setup = EvaluatorSetup.model(judge)
    profile = RoleProfile(
        LLMChat(lambda turns: "READY"), LLMChat(lambda turns: "READY"), setup, 5
    )
    assert run_preflight(profile) == 1
    report = json.loads(capsys.readouterr().out)
    assert "ModelVerdictError" in report["roles"][2]["error"]
    assert (
        setup.evaluator.evaluate(Message("Different base"), Message("Different result"))
        is False
    )
    assert len(judge.history()) == 2
