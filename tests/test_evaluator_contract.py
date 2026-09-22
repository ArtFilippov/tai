"""One-operation extensions through the actual FITD/readiness/series paths."""

import json
from pathlib import Path
from threading import Event
from typing import cast

import pytest
from evaluation_fakes import ScriptedModel
from journal_assertions import at, evaluation_of, mapping, released_worker, result_of, text

from tai import (
    EvaluatorSetup,
    ChatTurn,
    FITDExperiment,
    FITDUser,
    LLMChat,
    Message,
    UserState,
)
from app.configuration import RoleProfile
from app.preflight import run_preflight
from app.series import run_alert_series
from core.evaluation.evaluator import EvaluationChat


def participants(repeats: int = 1) -> tuple[LLMChat, LLMChat]:
    return (
        LLMChat(ScriptedModel(["One", "Two", "Three", "Four", "Presented"] * repeats)),
        LLMChat(ScriptedModel((["Initial", "L1", "L2", "L3"] + ["yes"] * 5) * repeats)),
    )


@pytest.mark.parametrize("user_type", [FITDUser, FITDExperiment])
def test_one_operation_receives_exact_messages_across_independent_runs(
    user_type: type[FITDUser] | type[FITDExperiment],
):
    pairs: list[tuple[Message, Message]] = []

    class CustomEvaluator:
        __slots__ = ()

        def evaluate(self, base: Message, result: Message) -> bool:
            pairs.append((base, result))
            return base.to_string() == "Second base"

    evaluator = CustomEvaluator()
    target, attacker = participants(3)
    bases = [Message("First base"), Message("Second base")]
    users: list[FITDUser | FITDExperiment] = []
    for base in bases:
        user = user_type(target, attacker, base, evaluator, max_calls_per_conv=5)
        user.start()
        assert user.wait(10)
        assert user.status().state is UserState.COMPLETED, user.status().reason
        assert pairs[-1][0] is base and pairs[-1][1] is user.result()
        assert evaluation_of(user).success is (base is bases[1])
        users.append(user)
    old = users[-1].snapshot()
    users[-1].start()
    assert users[-1].wait(10)
    assert users[-1].result() is not old.result
    assert evaluation_of(users[-1]).result_id == result_of(users[-1]).id
    assert len(pairs) == 3
    if user_type is FITDExperiment:
        for user in users:
            assert isinstance(user, FITDExperiment)
            assert_independent_assessments(user)
        assert len(pairs) == 3


def assert_independent_assessments(user: FITDExperiment) -> None:
    for journal in user.journals:
        settings = journal.settings()
        mapping(at(settings, "base"))["text"] = "Edited diagnostic"
        starts = [mapping(e.data) for e in journal.events() if e.kind == "evaluation_started"]
        ends = [mapping(e.data) for e in journal.events() if e.kind == "evaluation_completed"]
        assert len(starts) == len(ends) == 1
        assert starts[0]["evaluation_id"] == ends[0]["evaluation_id"]
        assert starts[0]["base_id"] == ends[0]["base_id"] == user.base.id
        assert at(starts[0], "result", "id") == ends[0]["result_id"]
        assert at(journal.settings(), "base", "text") == user.base.to_string()


@pytest.mark.parametrize(
    "verdict", [True, False, None, 0, 1, "false", [], RuntimeError("unavailable")]
)
def test_preflight_and_fitd_enforce_the_same_custom_evaluator_contract(
    verdict: object, capsys: pytest.CaptureFixture[str],
):
    pairs: list[tuple[Message, Message]] = []

    class CustomEvaluator:
        def evaluate(self, base: Message, result: Message) -> bool:
            pairs.append((base, result))
            if isinstance(verdict, Exception):
                raise verdict
            return cast(bool, verdict)  # Deliberately violate the runtime extension contract.

    evaluator = CustomEvaluator()
    profile = RoleProfile(
        LLMChat(lambda turns: "READY"),
        LLMChat(lambda turns: "READY"),
        EvaluatorSetup(evaluator),
        5,
    )
    valid = isinstance(verdict, bool)
    assert run_preflight(profile) == (0 if valid else 1)
    assert_preflight_assessment(json.loads(capsys.readouterr().out), verdict)
    assert pairs[0][0].to_string() == "Name the capital of France."
    user = FITDExperiment(
        *participants(), Message("Actual base"), evaluator, max_calls_per_conv=5
    )
    user.start()
    assert user.wait(10)
    assert user.status().state is (UserState.COMPLETED if valid else UserState.FAILED)
    assert len(pairs) == 2
    assert pairs[1][0] is user.base and pairs[1][1] is user.result()
    if valid:
        assert evaluation_of(user).success is verdict
    else:
        assert user.status().evaluation is None
        failure = next(
            mapping(e.data) for e in user.journal.events() if e.kind == "evaluation_failed"
        )
        assert failure["base_id"] == user.base.id
        assert failure["result_id"] == result_of(user).id


def assert_preflight_assessment(report: object, verdict: object) -> None:
    assessment = mapping(at(report, "roles", 2))
    assert assessment["ok"] is isinstance(verdict, bool)
    if isinstance(verdict, bool):
        assert assessment["success"] is verdict
    else:
        assert "success" not in assessment
        assert (
            "unavailable" if isinstance(verdict, Exception) else "must return bool"
        ) in text(assessment["error"])


def mutate_assessment_diagnostics(user: FITDExperiment, setup: EvaluatorSetup) -> None:
    event = next(e for e in user.journal.events() if e.kind == "evaluation_started")
    copy = event.data
    mapping(at(copy, "base"))["text"] = "Edited base"
    mapping(at(copy, "result"))["text"] = "Edited result"
    setup.settings()["expected"] = "Edited criterion"
    mapping(at(user.journal.settings(), "evaluator"))["expected"] = "Edited criterion"


def test_stop_waits_for_custom_assessment_and_diagnostic_mutation_cannot_change_input():
    entered, release = Event(), Event()
    pairs: list[tuple[Message, Message]] = []
    configuration = {"expected": "Presented"}

    class CustomEvaluator:
        def evaluate(self, base: Message, result: Message) -> bool:
            pairs.append((base, result))
            entered.set()
            assert release.wait(10)
            assert base.to_string() == "Base"
            assert result.to_string() == configuration["expected"] == "Presented"
            return False

    setup = EvaluatorSetup(CustomEvaluator(), configuration=lambda: configuration)
    user = FITDExperiment(*participants(), Message("Base"), setup)
    user.start()
    with released_worker(user, release):
        assert entered.wait(10)
        result = result_of(user)
        mutate_assessment_diagnostics(user, setup)
        user.stop()
        assert user.status().state is UserState.STOPPING
        assert not user.wait(0)
    assert user.status().state is UserState.STOPPED
    assert evaluation_of(user).result_id == result.id
    assert evaluation_of(user).success is False
    assert pairs == [(user.base, result)]
    events = user.journal.events()
    stop = next(e.sequence for e in events if e.kind == "stop_requested")
    assert not any(
        e.kind in {"call_started", "evaluation_started"} and e.sequence > stop
        for e in events
    )


def test_series_uses_the_assembled_one_operation_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    pairs: list[tuple[Message, Message]] = []

    class CustomEvaluator:
        def evaluate(self, base: Message, result: Message) -> bool:
            pairs.append((base, result))
            return True

    def factory() -> RoleProfile:
        return RoleProfile(*participants(1), EvaluatorSetup(CustomEvaluator()), 20)

    profile = factory()
    def fresh(_profile: RoleProfile) -> RoleProfile:
        return factory()

    monkeypatch.setattr(RoleProfile, "fresh", fresh)

    def load_profile(*_args: object, **_kwargs: object) -> RoleProfile:
        return profile

    monkeypatch.setattr("app.series.load_profile", load_profile)
    series = run_alert_series(
        Path("examples/alert-manifest.json"),
        Path("examples/fitd-alert-qwen.toml"),
        tmp_path,
        taxonomy_path=Path("ALERT_taxonomy.md"),
    )
    assert not series.exit_code
    index = json.loads((series.directory / "index.json").read_text())
    assert len(pairs) == len(index["entries"]) == 32
    assert len({base.id for base, _ in pairs}) == 32
    assert len({result.id for _, result in pairs}) == 32
    by_base = {base.id: (base, result) for base, result in pairs}
    for entry in index["entries"]:
        base, result = by_base[entry["base_id"]]
        assert entry["base_id"] == base.id
        assert entry["evaluation"] == {"result_id": result.id, "success": True}


@pytest.mark.parametrize("stop", [False, True])
def test_custom_chat_evaluator_is_observed_and_owned_without_class_specific_io(stop: bool):
    entered, release = Event(), Event()
    requests: list[tuple[ChatTurn, ...]] = []

    def generate(turns: tuple[ChatTurn, ...]) -> str:
        requests.append(turns)
        if len(requests) == 1:
            entered.set()
            assert release.wait(10)
        return "allow"

    class TwoExchangeEvaluator:
        def __init__(self, chat: EvaluationChat) -> None:
            super().__init__()
            self.chat = chat

        def evaluate(self, base: Message, result: Message) -> bool:
            self.chat.reset()
            self.chat.send(base)
            return self.chat.send(result).to_string() == "allow"

    judge = LLMChat(generate, system="Independent classifier")
    setup = EvaluatorSetup.with_chat(judge, TwoExchangeEvaluator)
    user = FITDExperiment(*participants(), Message("Actual base"), setup)
    contender = FITDUser(*participants(), Message("Other"), setup)
    user.start()
    with released_worker(user, release):
        assert entered.wait(10)
        with pytest.raises(RuntimeError, match="active execution"):
            contender.start()
        if stop:
            user.stop()
    assert user.status().state is (UserState.STOPPED if stop else UserState.COMPLETED)
    assert evaluation_of(user).success is True
    assert requests[0][-1].message is user.base
    assert requests[1][-1].message is user.result()
    assert_custom_chat_events(user)
    assert judge._execution.owner is None


def assert_custom_chat_events(user: FITDExperiment) -> None:
    starts = [mapping(e.data) for e in user.journal.events() if e.kind == "evaluation_started"]
    calls = [
        mapping(e.data)
        for e in user.journal.events()
        if e.kind == "call_completed" and at(e.data, "role") == "Evaluator"
    ]
    assert len(starts) == 1 and len(calls) == 2
    assert all(c["evaluation_id"] == starts[0]["evaluation_id"] for c in calls)
    assert all(c["result_id"] == result_of(user).id for c in calls)
    assert user.status().progress.evaluator_calls == 1


def test_failed_preflight_does_not_leave_observation_bound_to_shared_evaluator(
    capsys: pytest.CaptureFixture[str],
):
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
