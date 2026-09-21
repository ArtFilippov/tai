"""Observe the real pinned FITD through operation arguments/returns alone."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fitd_observation import Observations, OperationValues, install_observers, puzzle

from attacks.fitd.execution import Execution
from core.runs.operations import BranchDecision, Cause, Context, EvaluationResult, ResultChange, TargetExchange, WorkerOutcome
from tai import FITDExperiment as FITDUser, LLMChat, Message, UserState


@pytest.fixture
def observed(monkeypatch: pytest.MonkeyPatch) -> Observations:
    return install_observers(monkeypatch)


@pytest.mark.parametrize(
    ("targets", "continuations", "branches", "history_change"),
    [
        (["Same", "Same"], ["yes", "yes"], ["continue", "continue"], "unchanged"),
        (["Opening", "I cannot help with that"], ["yes", "Rephrased"], ["continue", "rephrase"], "restored"),
        (["I cannot help with that"], ["Safer"], ["restart"], "reset"),
        (["Opening", "Off topic"], ["yes", "no"], ["continue", "realign"], "unchanged"),
        (["Opening", "Off topic", "Correction"], ["yes", "no", "no"], ["continue", "realign", "continue"], "unchanged"),
    ],
)
def test_native_continuations_drive_observable_operations_on_the_final_round(
    tmp_path: Path, observed: Observations, targets: list[str | Exception],
    continuations: list[str], branches: list[str], history_change: str,
):
    user, target, _ = puzzle(tmp_path, targets, ["Initial", "L1", "L2", "L3", *continuations], len(targets))
    user.start()
    # Worker completion is observed without User.wait().
    assert observed.finished.wait(10)
    values = observed.values
    _assert_worker_outcome(user, values)
    selections = [selection.decision for selection in values["select_continuation"]]
    assert [selection.branch for selection in selections] == branches
    assert observed.native["generate_next_turn_attempt"] == len(selections)
    assert observed.native["_is_refusal"] == len(values["check_refusal"])
    assert observed.native["_is_response_aligned"] == len(values["check_alignment"])
    assert observed.native["_is_realign_prompt"] == continuations.count("no")
    assert len(values["exchange"]) == len(targets)
    for exchange, selection in zip(values["exchange"], selections, strict=True):
        _assert_selected_exchange(values, exchange, selection)
    for earlier, later in zip(selections[:-1], values["exchange"][1:], strict=True):
        assert earlier.next_attempt_id == later.exchange.context.cause.attempt_id
        assert earlier.next_request is later.exchange.request
    _assert_final_synchronization(target, values, history_change)
    _assert_retained_material(target, values, history_change, continuations[-1])
    if targets == ["Same", "Same"]:
        changes = values["accept"]
        assert [change.kind for change in changes] == ["accepted", "replaced"]
        assert changes[-1].previous is changes[0].current
        assert changes[-1].current is not None and changes[-1].previous is not None
        assert changes[-1].current.id != changes[-1].previous.id


def _assert_worker_outcome(user: FITDUser, values: OperationValues):
    outcome = values["_run"][0]
    assert outcome.status.state is UserState.COMPLETED, outcome.status.reason
    assert outcome.status.reason == "round_limit"
    assert outcome.error is None
    assert outcome.released is values["_release_resources"][0]
    assert outcome.released is not None
    assert outcome.released.roles == ("Target", "Attacker")
    assert values["start"][0].run_id == outcome.status.run_id
    assert outcome.result is user.result()


def _assert_selected_exchange(values: OperationValues, exchange: TargetExchange, selection: BranchDecision):
    assert exchange.point.checkpoint in values["checkpoint"]
    assert exchange.point in values["save"]
    assert exchange.exchange in values["_send"]
    assert selection.point is exchange.point
    assert selection.context.run_id == values["_run"][0].status.run_id
    assert selection.context.conversation_id == exchange.exchange.context.conversation_id
    assert selection.context.cause.attempt_id == exchange.exchange.context.cause.attempt_id
    assert selection.context.cause.id == selection.id
    for check in selection.checks:
        assert check.context == selection.context
        assert check.request is exchange.exchange.request
        assert check.response is exchange.exchange.response
        assert check.checkpoint_id == exchange.point.checkpoint.id
        if check.kind == "alignment":
            assert check.judgment in values["_send"]
            assert check.judgment is not None
            assert check.judgment.context.cause == selection.context.cause
            assert check.judgment.context.role == "Attacker"
    if selection.branch == "continue":
        assert selection.next_levels_completed == selection.levels_completed + 1
    else:
        assert selection.next_levels_completed == selection.levels_completed


def _assert_final_synchronization(target: LLMChat, values: OperationValues, history_change: str):
    outcome = values["_run"][0]
    final = values["select_continuation"][-1].decision
    synchronization = values["synchronize"][-1]
    assert synchronization.conversation.kind == history_change
    assert synchronization.conversation.context.cause.id == final.id
    assert synchronization.conversation.context.cause.reason == final.branch
    assert synchronization.conversation.context.conversation_id == target.conversation_id
    assert synchronization.before.checkpoint == final.point.checkpoint
    assert synchronization.after.result is outcome.result
    assert synchronization.result.current is outcome.result
    assert synchronization.result.kind == "unchanged"
    assert final.next_attempt_id not in {exchange.exchange.context.cause.attempt_id for exchange in values["exchange"]}
    assert final.next_request is not None
    assert all(final.next_request is not item.exchange.request for item in values["exchange"])


def _assert_retained_material(target: LLMChat, values: OperationValues, history_change: str, continuation: str):
    outcome = values["_run"][0]
    final = values["select_continuation"][-1].decision
    synchronization = values["synchronize"][-1]
    if history_change == "restored":
        assert synchronization.after.response is values["exchange"][0].exchange.response
        assert target.history()[-1].message is outcome.result
        assert values["restore"][0].checkpoint is synchronization.after.checkpoint
    elif history_change == "reset":
        assert synchronization.after.request is synchronization.after.response is None
        assert outcome.result is None
        assert len(target.history()) == 1
        with pytest.raises(ValueError, match="stale"):
            target.restore(synchronization.before.checkpoint.token)
    elif final.branch == "realign":
        assert target.history()[-1].message is final.point.response
        assert outcome.result is not final.point.response
    elif continuation == "no":
        assert final.checks[-1].value is False
        assert outcome.result is final.point.response


def test_evaluation_commit_and_replacement_refer_to_the_exact_presented_message(tmp_path: Path, observed: Observations):
    user, _, _ = puzzle(tmp_path, ["One", "Two", "Three", "Four", "Negative", "CONTROL"], ["Initial", "L1", "L2", "L3"] + ["yes"] * 6, 6)
    user.start()
    assert observed.finished.wait(10)
    values = observed.values
    evaluations = values["record_evaluation"]
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
    [("I cannot help with that", "Rephrased", "rephrase"), ("Unrelated CONTROL", "no", "realign")],
)
def test_rejected_material_preserves_the_exact_retained_result_and_score(
    tmp_path: Path, observed: Observations, response: str, judgment: str, branch: str,
):
    user, _, _ = puzzle(tmp_path, ["One", "Two", "Three", "Four", "Negative", response], ["Initial", "L1", "L2", "L3"] + ["yes"] * 5 + [judgment], 6)
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
    tmp_path: Path, observed: Observations, monkeypatch: pytest.MonkeyPatch,
):
    record = Execution.record_evaluation
    changes: list[ResultChange] = []

    def rollback(execution: Execution, result: Message, success: bool, context: Context, checkpoint_id: str) -> EvaluationResult:
        evaluated = record(execution, result, success, context, checkpoint_id)
        earlier = observed.values["accept"][0]
        cause = replace(context, cause=Cause("restore", context.cause.attempt_id))
        changes.append(execution.publish(earlier.current, cause, earlier.checkpoint_id))
        with pytest.raises(RuntimeError, match="unpresented Message"):
            record(execution, result, True, context, checkpoint_id)
        changes.append(execution.publish(None, cause, earlier.checkpoint_id))
        assert not (tmp_path / context.run_id / "result.txt").exists()
        changes.append(execution.publish(None, cause, earlier.checkpoint_id))
        changes.append(execution.publish(result, context, checkpoint_id))
        return evaluated

    monkeypatch.setattr(Execution, "record_evaluation", rollback)
    user, _, _ = puzzle(tmp_path, ["One", "Two", "Three", "Four", "Negative"], ["Initial", "L1", "L2", "L3"] + ["yes"] * 5, 5)
    user.start()
    assert observed.finished.wait(10)
    assert user.status().state is UserState.COMPLETED, user.status().reason
    _assert_rollback(user, changes)


def _assert_rollback(user: FITDUser, changes: list[ResultChange]):
    assert [change.kind for change in changes] == ["replaced", "withdrawn", "unchanged", "accepted"]
    replaced, withdrawn, unchanged, accepted = changes
    assert replaced.previous is accepted.current
    assert replaced.previous_evaluation is not None and replaced.previous is not None
    assert replaced.previous_evaluation.result_id == replaced.previous.id
    assert replaced.evaluation is None
    assert withdrawn.previous is replaced.current
    assert withdrawn.current is unchanged.current is accepted.previous is None
    assert user.result() is accepted.current
    assert user.status().evaluation is None
    journal_changes = [
        json.loads(event.dump())["data"] for event in user.journal.events()
        if event.kind in {"result_published", "result_withdrawn", "result_unchanged"}
    ]
    withdrawal = next(change for change in journal_changes if change["change"] == "withdrawn")
    assert withdrawn.previous is not None
    assert withdrawal["previous"]["id"] == withdrawn.previous.id
    assert withdrawal["result"] is None
    assert withdrawal["checkpoint_id"] == withdrawn.checkpoint_id
    assert withdrawal["context"]["cause"]["reason"] == "restore"


def test_initial_exhaustion_exposes_attacker_refusals_and_the_worker_outcome(tmp_path: Path, observed: Observations):
    user, _, _ = puzzle(tmp_path, [], ["I cannot help with that"] * 3, 1)
    user.start()
    assert observed.finished.wait(10)
    values = observed.values
    assert not values["select_continuation"]
    assert observed.native["_is_refusal"] == 3
    for decision, exchange in zip(values["check_refusal"], values["_send"], strict=True):
        assert decision.value is True
        assert decision.response is exchange.response
        assert decision.context == exchange.context
        assert decision.context.cause.attempt_id is None
        assert decision.context.role == "Attacker"
    outcome = values["_run"][0]
    assert outcome.status.state is UserState.COMPLETED
    assert outcome.status.reason == "strategy_exhausted"
    assert outcome.result is None


def test_operation_contexts_and_checkpoints_are_independent_across_repeated_runs(tmp_path: Path, observed: Observations):
    user, target, _ = puzzle(tmp_path, ["Same", "Same"], ["Initial", "L1", "L2", "L3", "yes"] * 2, 1)
    user.start()
    assert observed.finished.wait(10)
    assert user.wait(5)
    first: WorkerOutcome = observed.values["_run"][0]
    first_point = observed.values["exchange"][0].point
    observed.finished.clear()
    user.start()
    assert observed.finished.wait(10)
    second = observed.values["_run"][1]
    second_exchange = observed.values["exchange"][1]
    assert first.status.state is second.status.state is UserState.COMPLETED
    assert first.status.run_id != second.status.run_id
    assert first.result is not None and second.result is not None
    assert first.result.id != second.result.id
    assert second_exchange.exchange.context.run_id == second.status.run_id
    assert second_exchange.point.checkpoint.context.conversation_id != first_point.checkpoint.context.conversation_id
    with pytest.raises(ValueError, match="stale"):
        target.restore(first_point.checkpoint.token)
    assert user.result() is second.result
