"""Observe the real pinned FITD through operation arguments/returns alone."""

from collections import Counter, defaultdict
from dataclasses import replace
from functools import wraps
from pathlib import Path
from threading import Event

import pytest
from garak.exception import GarakException
from garak.probes.fitd import FITD

from tai import ContainsTextEvaluator, FITDUser, LLMChat, Message, UserState
from tai._fitd import _FITD, _Target
from tai._operations import Cause
from tai.fitd import _Execution


class Observations:
    def __init__(self):
        self.values = defaultdict(list)
        self.errors = defaultdict(list)
        self.native = Counter()
        self.finished = Event()

    def wrap(self, operation):
        @wraps(operation)
        def observe(receiver, *args, **kwargs):
            try:
                value = operation(receiver, *args, **kwargs)
            except Exception as exc:
                self.errors[operation.__name__].append(exc)
                raise
            self.values[operation.__name__].append(value)
            if operation.__name__ == "_run":
                self.finished.set()
            return value

        return observe


@pytest.fixture
def observed(monkeypatch):
    observations = Observations()
    for owner, methods in (
        (_FITD, ("check_refusal", "check_alignment", "select_continuation")),
        (_Target, ("save", "exchange", "synchronize", "accept")),
        (
            _Execution,
            (
                "start",
                "_send",
                "_checkpoint",
                "_restore",
                "_reset",
                "_publish",
                "_evaluate",
                "_record_evaluation",
                "_run",
                "_release_resources",
            ),
        ),
    ):
        for method in methods:
            monkeypatch.setattr(
                owner, method, observations.wrap(getattr(owner, method))
            )

    def count(operation):
        @wraps(operation)
        def call(*args, **kwargs):
            observations.native[operation.__name__] += 1
            return operation(*args, **kwargs)

        return call

    for method in (
        "_is_refusal",
        "_is_response_aligned",
        "_is_realign_prompt",
        "generate_next_turn_attempt",
    ):
        monkeypatch.setattr(FITD, method, count(getattr(FITD, method)))
    return observations


def scripted(responses):
    replies = iter(responses)

    def generate(turns):
        response = next(replies)
        if isinstance(response, Exception):
            raise response
        return response

    return generate


def puzzle(tmp_path, targets, attackers, rounds):
    target = LLMChat(scripted(targets), system="Puzzle rules")
    attacker = LLMChat(scripted(attackers), system="Build puzzle requests")
    user = FITDUser(
        target,
        attacker,
        Message("Solve the fictional puzzle"),
        ContainsTextEvaluator("CONTROL"),
        artifacts_dir=tmp_path,
        max_calls_per_conv=rounds,
    )
    return user, target, attacker


@pytest.mark.parametrize(
    ("targets", "continuations", "branches", "history_change"),
    [
        (["Same", "Same"], ["yes", "yes"], ["continue", "continue"], "unchanged"),
        (
            ["Opening", "I cannot help with that"],
            ["yes", "Rephrased"],
            ["continue", "rephrase"],
            "restored",
        ),
        (["I cannot help with that"], ["Safer"], ["restart"], "reset"),
        (["Opening", "Off topic"], ["yes", "no"], ["continue", "realign"], "unchanged"),
        (
            ["Opening", "Off topic", "Correction"],
            ["yes", "no", "no"],
            ["continue", "realign", "continue"],
            "unchanged",
        ),
    ],
)
def test_native_continuations_drive_observable_operations_on_the_final_round(
    tmp_path, observed, targets, continuations, branches, history_change
):
    user, target, _ = puzzle(
        tmp_path, targets, ["Initial", "L1", "L2", "L3", *continuations], len(targets)
    )
    user.start()
    # The wrapper sees worker completion without any call to User.wait().
    assert observed.finished.wait(10)
    values = observed.values
    outcome = values["_run"][0]
    assert outcome.status.state is UserState.COMPLETED, outcome.status.reason
    assert outcome.status.reason == "round_limit"
    assert outcome.error is None
    assert outcome.released is values["_release_resources"][0]
    assert outcome.released.roles == ("Target", "Attacker")
    assert values["start"][0].run_id == outcome.status.run_id
    assert outcome.result is user.result()

    selections = [selection.decision for selection in values["select_continuation"]]
    assert [selection.branch for selection in selections] == branches
    assert observed.native["generate_next_turn_attempt"] == len(selections)
    assert observed.native["_is_refusal"] == len(values["check_refusal"])
    assert observed.native["_is_response_aligned"] == len(values["check_alignment"])
    assert observed.native["_is_realign_prompt"] == continuations.count("no")
    assert len(values["exchange"]) == len(targets)

    for exchange, selection in zip(values["exchange"], selections, strict=True):
        assert exchange.point.checkpoint in values["_checkpoint"]
        assert exchange.point in values["save"]
        assert exchange.exchange in values["_send"]
        assert selection.point is exchange.point
        assert selection.context.run_id == outcome.status.run_id
        assert (
            selection.context.conversation_id
            == exchange.exchange.context.conversation_id
        )
        assert (
            selection.context.cause.attempt_id
            == exchange.exchange.context.cause.attempt_id
        )
        assert selection.context.cause.id == selection.id
        for check in selection.checks:
            assert check.context == selection.context
            assert check.request is exchange.exchange.request
            assert check.response is exchange.exchange.response
            assert check.checkpoint_id == exchange.point.checkpoint.id
            if check.kind == "alignment":
                assert check.judgment in values["_send"]
                assert check.judgment.context.cause == selection.context.cause
                assert check.judgment.context.role == "Attacker"
        if selection.branch == "continue":
            assert selection.next_levels_completed == selection.levels_completed + 1
        else:
            assert selection.next_levels_completed == selection.levels_completed

    for earlier, later in zip(selections[:-1], values["exchange"][1:], strict=True):
        assert earlier.next_attempt_id == later.exchange.context.cause.attempt_id
        assert earlier.next_request is later.exchange.request

    final = selections[-1]
    synchronization = values["synchronize"][-1]
    assert synchronization.conversation.kind == history_change
    assert synchronization.conversation.context.cause.id == final.id
    assert synchronization.conversation.context.cause.reason == final.branch
    assert (
        synchronization.conversation.context.conversation_id == target.conversation_id
    )
    assert synchronization.before.checkpoint == final.point.checkpoint
    assert synchronization.after.result is outcome.result
    assert synchronization.result.current is outcome.result
    assert synchronization.result.kind == "unchanged"
    # A prepared continuation is observable but is never sent beyond the budget.
    assert final.next_attempt_id not in {
        exchange.exchange.context.cause.attempt_id for exchange in values["exchange"]
    }
    assert final.next_request is not None
    assert all(
        final.next_request is not item.exchange.request for item in values["exchange"]
    )
    if history_change == "restored":
        assert synchronization.after.response is values["exchange"][0].exchange.response
        assert target.history()[-1].message is outcome.result
        assert values["_restore"][0].checkpoint is synchronization.after.checkpoint
    elif history_change == "reset":
        assert synchronization.after.request is synchronization.after.response is None
        assert outcome.result is None
        assert len(target.history()) == 1
        with pytest.raises(ValueError, match="stale"):
            target.restore(synchronization.before.checkpoint.token)
    elif branches[-1] == "realign":
        assert target.history()[-1].message is final.point.response
        assert outcome.result is not final.point.response
    elif continuations[-1] == "no":
        # Negative alignment does not imply realignment: garak accepts a correction.
        assert final.checks[-1].value is False
        assert outcome.result is final.point.response
    if targets == ["Same", "Same"]:
        changes = values["accept"]
        assert [change.kind for change in changes] == ["accepted", "replaced"]
        assert changes[-1].previous is changes[0].current
        assert changes[-1].current.id != changes[-1].previous.id


def test_evaluation_commit_and_replacement_refer_to_the_exact_presented_message(
    tmp_path, observed
):
    user, _, _ = puzzle(
        tmp_path,
        ["One", "Two", "Three", "Four", "Negative", "CONTROL"],
        ["Initial", "L1", "L2", "L3"] + ["yes"] * 6,
        6,
    )
    user.start()
    assert observed.finished.wait(10)
    values = observed.values
    evaluations = values["_record_evaluation"]
    assert evaluations == values["_evaluate"]
    assert [value.evaluation.success for value in evaluations] == [False, True]
    first, second = evaluations
    last_change = values["accept"][-1]
    assert last_change.kind == "replaced"
    assert last_change.previous is first.result
    assert last_change.previous_evaluation is first.evaluation
    assert last_change.evaluation is None
    assert last_change.current is second.result is user.result()
    assert second.checkpoint_id == last_change.checkpoint_id
    assert second.context == last_change.context
    assert user.status().evaluation is second.evaluation
    assert values["_run"][0].status.reason == "evaluator_success"


@pytest.mark.parametrize(
    ("response", "judgment", "branch"),
    [
        ("I cannot help with that", "Rephrased", "rephrase"),
        ("Unrelated CONTROL", "no", "realign"),
    ],
)
def test_rejected_material_preserves_the_exact_retained_result_and_score(
    tmp_path, observed, response, judgment, branch
):
    user, _, _ = puzzle(
        tmp_path,
        ["One", "Two", "Three", "Four", "Negative", response],
        ["Initial", "L1", "L2", "L3"] + ["yes"] * 5 + [judgment],
        6,
    )
    user.start()
    assert observed.finished.wait(10)
    values = observed.values
    (evaluated,) = values["_evaluate"]
    final = values["select_continuation"][-1].decision
    assert final.branch == branch
    synchronization = values["synchronize"][-1]
    assert synchronization.result.kind == "unchanged"
    assert synchronization.result.current is evaluated.result is user.result()
    assert synchronization.result.evaluation is evaluated.evaluation
    assert user.status().evaluation is evaluated.evaluation
    assert final.point.response is not user.result()


def test_result_rollback_withdrawal_and_stale_verdict_contract(
    tmp_path, observed, monkeypatch
):
    # FITD's pinned branches only rewind a rejected exchange. Exercise the more
    # general result-operation contract too, using real accepted Messages and a
    # completed evaluation from the live worker, without reading its state.
    record = _Execution._record_evaluation
    changes = []

    def rollback(execution, result, success, context, checkpoint_id):
        evaluated = record(execution, result, success, context, checkpoint_id)
        earlier = observed.values["accept"][0]
        cause = replace(context, cause=Cause("restore", context.cause.attempt_id))
        changes.append(
            execution._publish(earlier.current, cause, earlier.checkpoint_id)
        )
        with pytest.raises(RuntimeError, match="unpresented Message"):
            record(execution, result, True, context, checkpoint_id)
        changes.append(execution._publish(None, cause, earlier.checkpoint_id))
        assert not (tmp_path / context.run_id / "result.txt").exists()
        changes.append(execution._publish(None, cause, earlier.checkpoint_id))
        changes.append(execution._publish(result, context, checkpoint_id))
        return evaluated

    monkeypatch.setattr(_Execution, "_record_evaluation", rollback)
    user, _, _ = puzzle(
        tmp_path,
        ["One", "Two", "Three", "Four", "Negative"],
        ["Initial", "L1", "L2", "L3"] + ["yes"] * 5,
        5,
    )
    user.start()
    assert observed.finished.wait(10)
    assert user.status().state is UserState.COMPLETED, user.status().reason
    assert [change.kind for change in changes] == [
        "replaced",
        "withdrawn",
        "unchanged",
        "accepted",
    ]
    replaced, withdrawn, unchanged, accepted = changes
    assert replaced.previous is accepted.current
    assert replaced.previous_evaluation.result_id == replaced.previous.id
    assert replaced.evaluation is None
    assert withdrawn.previous is replaced.current
    assert withdrawn.current is unchanged.current is accepted.previous is None
    assert user.result() is accepted.current
    assert user.status().evaluation is None


def test_initial_exhaustion_exposes_attacker_refusals_and_the_worker_outcome(
    tmp_path, observed
):
    user, _, _ = puzzle(tmp_path, [], ["I cannot help with that"] * 3, 1)
    user.start()
    assert observed.finished.wait(10)
    values = observed.values
    assert values["select_continuation"] == []
    assert observed.native["_is_refusal"] == 3
    for decision, exchange in zip(
        values["check_refusal"], values["_send"], strict=True
    ):
        assert decision.value is True
        assert decision.response is exchange.response
        assert decision.context == exchange.context
        assert decision.context.cause.attempt_id is None
        assert decision.context.role == "Attacker"
    outcome = values["_run"][0]
    assert outcome.status.state is UserState.COMPLETED
    assert outcome.status.reason == "strategy_exhausted"
    assert outcome.result is None


@pytest.mark.parametrize("failure", ["provider", "decision", "artifacts", "setup"])
def test_worker_returns_actual_failures_including_errors_caught_by_garak(
    tmp_path, observed, monkeypatch, failure
):
    error = GarakException("native boundary failed")
    responses = [error] if failure == "provider" else ["Opening"]
    if failure == "decision":

        def fail(*args):
            raise error

        monkeypatch.setattr(FITD, "_is_refusal", fail)
    elif failure == "artifacts":
        write = Path.write_text

        def fail_metadata(path, *args, **kwargs):
            if path.name == "metadata.json":
                raise error
            return write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_metadata)
    elif failure == "setup":
        mkdir = Path.mkdir

        def fail_directory(path, *args, **kwargs):
            if path.parent == tmp_path:
                raise error
            return mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail_directory)
    user, target, attacker = puzzle(
        tmp_path, responses, ["Initial", "L1", "L2", "L3", "yes"], 1
    )
    user.start()
    assert observed.finished.wait(10)
    outcome = observed.values["_run"][0]
    assert outcome.status.state is UserState.FAILED
    assert outcome.status.reason == "GarakException: native boundary failed"
    assert outcome.error is error
    assert outcome.released.roles == ("Target", "Attacker")
    # A fresh User can reserve these participants after the observed release.
    next_user = FITDUser(
        target,
        attacker,
        Message("Another puzzle"),
        ContainsTextEvaluator("CONTROL"),
        artifacts_dir=tmp_path,
        max_calls_per_conv=1,
    )
    next_user.start()
    assert next_user.wait(10)


@pytest.mark.parametrize("stop", [False, True])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_start_return_is_distinct_from_work_and_resource_cleanup(
    tmp_path, observed, monkeypatch, stop, cleanup_failure
):
    import tai.fitd as implementation

    entered, allow_response = Event(), Event()
    releasing, allow_release = Event(), Event()
    release_chats = implementation._release_chats

    def target_model(turns):
        entered.set()
        assert allow_response.wait(5)
        return "Opening"

    def cleanup(chats, owner):
        releasing.set()
        assert allow_release.wait(5)
        release_chats(chats, owner)
        if cleanup_failure:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(implementation, "_release_chats", cleanup)
    user = FITDUser(
        LLMChat(target_model),
        LLMChat(scripted(["Initial", "L1", "L2", "L3", "yes"])),
        Message("Puzzle"),
        ContainsTextEvaluator("CONTROL"),
        artifacts_dir=tmp_path,
        max_calls_per_conv=1,
    )
    try:
        assert user.start() is None
        assert entered.wait(5)
        assert observed.values["start"][0].run_id == user.status().run_id
        assert not observed.finished.is_set()
        if stop:
            user.stop()
        allow_response.set()
        assert releasing.wait(5)
        assert user.status().state is (
            UserState.STOPPED if stop else UserState.COMPLETED
        )
        assert not observed.finished.is_set()
        assert observed.values["_release_resources"] == []
        terminal = user.status()
        user.stop()
        assert user.status() == terminal
        assert not user.wait(0)
        with pytest.raises(RuntimeError, match="active run"):
            user.start()
    finally:
        allow_response.set()
        allow_release.set()
    assert observed.finished.wait(10)
    outcome = observed.values["_run"][0]
    if cleanup_failure:
        assert outcome.status.state is UserState.FAILED
        assert outcome.status.reason == "RuntimeError: cleanup failed"
        assert outcome.error is observed.errors["_release_resources"][0]
        assert outcome.released is None
        return
    assert outcome.released is not None
    assert outcome.status.state is (UserState.STOPPED if stop else UserState.COMPLETED)
    if stop:
        assert outcome.status.reason == "user_stop"
        assert not observed.errors["_send"]
        assert (
            observed.values["exchange"][-1].exchange.response.to_string() == "Opening"
        )


def test_operation_contexts_and_checkpoints_are_independent_across_repeated_runs(
    tmp_path, observed
):
    user, target, _ = puzzle(
        tmp_path,
        ["Same", "Same"],
        ["Initial", "L1", "L2", "L3", "yes"] * 2,
        1,
    )
    user.start()
    assert observed.finished.wait(10)
    assert user.wait(5)
    first = observed.values["_run"][0]
    first_point = observed.values["exchange"][0].point
    observed.finished.clear()
    user.start()
    assert observed.finished.wait(10)
    second = observed.values["_run"][1]
    second_exchange = observed.values["exchange"][1]
    assert first.status.state is second.status.state is UserState.COMPLETED
    assert first.status.run_id != second.status.run_id
    assert first.result.id != second.result.id
    assert second_exchange.exchange.context.run_id == second.status.run_id
    assert second_exchange.point.checkpoint.context.conversation_id != (
        first_point.checkpoint.context.conversation_id
    )
    with pytest.raises(ValueError, match="stale"):
        target.restore(first_point.checkpoint.token)
    assert user.result() is second.result
