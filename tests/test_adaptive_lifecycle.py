"""Stopping, failures, cleanup and external persistence use shared execution."""

from pathlib import Path
from types import ModuleType

import pytest
from garak.probes import adaptive_attacks as native
from evaluation_fakes import HeldResponse, ScriptedModel
from fitd_support import releasing
from test_fitd_stop_boundaries import hold_boundary

import attacks.adaptive_attacks.garak_adapter as adapter
from attacks.shared import execution
from tai import (
    AdaptiveAttacksExperiment, ContainsTextEvaluator, EvaluatorSetup, LLMChat,
    Message, RunEvent, RunJournal, UserState, capture_run, save_run,
)


@pytest.mark.parametrize("role", ["Target", "Evaluator"])
@pytest.mark.parametrize("success", [False, True])
def test_stop_finishes_admitted_calls_without_publishing_an_unaccepted_exchange(role: str, success: bool, tmp_path: Path) -> None:
    held = HeldResponse('{"success": ' + str(success).lower() + '}')
    target = LLMChat.cancellable(ScriptedModel([held if role == "Target" else "Presented"]))
    evaluator_model = ScriptedModel([held])
    user = AdaptiveAttacksExperiment(target, Message("Base"), EvaluatorSetup.model(LLMChat.cancellable(evaluator_model)), journal_dir=tmp_path)
    user.start()
    with releasing(user, held.release):
        assert held.entered.wait(5)
        user.stop()
        user.stop()
        assert user.status().state is UserState.STOPPING
        assert not user.wait(0)
        assert held.cancellation is not None and not held.cancellation.is_set()
    assert user.status().state is UserState.STOPPED
    assert user.status().reason == "user_stop"
    result, evaluation = user.result(), user.status().evaluation
    if role == "Target":
        assert result is evaluation is None
        assert not evaluator_model.requests
    else:
        assert result is not None and result.text == "Presented"
        assert evaluation is not None and evaluation.result_id == result.id and evaluation.success is success
    record = capture_run(user, journal=user.journal)
    assert not list(tmp_path.glob("*/metadata.json"))
    directory = save_run(record, tmp_path)
    assert (directory / "metadata.json").exists()
    assert (directory / "result.txt").exists() == (role == "Evaluator")
    assert any(event.kind == "resources_released" for event in user.journal.events())


@pytest.mark.parametrize("failure", ["target", "evaluator", "resource", "resource_import", "journal", "cleanup"])
def test_failures_are_worker_errors_and_participants_are_released(failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = ScriptedModel([RuntimeError("target failed") if failure == "target" else "Presented"])
    target = LLMChat(model)
    evaluator = EvaluatorSetup.model(LLMChat(ScriptedModel([RuntimeError("evaluator failed") if failure == "evaluator" else '{"success": true}'])))
    user = AdaptiveAttacksExperiment(target, Message("Base"), evaluator, journal_dir=tmp_path)
    with monkeypatch.context() as patch:
        _inject_failure(patch, failure, tmp_path)
        user.start()
        assert user.wait(10)
    assert user.status().state is UserState.FAILED, user.status().reason
    assert user.journal.finished
    record = capture_run(user, journal=user.journal)
    assert record.snapshot.status.state is UserState.FAILED
    assert save_run(record, tmp_path).is_dir()
    model.responses.append("yes")
    following = AdaptiveAttacksExperiment(target, Message("New base"), ContainsTextEvaluator("yes"))
    following.start()
    assert following.wait(10)
    assert following.status().reason == "evaluator_success"


def _inject_failure(patch: pytest.MonkeyPatch, failure: str, root: Path) -> None:
    if failure == "resource":
        patch.setattr(native, "SUFFIXES_PATH", root / "missing-suffixes.txt")
    elif failure == "resource_import":
        patch.setattr(adapter, "import_module", _fail_resource_import)
    elif failure == "journal":
        _fail_journal(patch)
    elif failure == "cleanup":
        _fail_cleanup(patch)


def _fail_resource_import(name: str) -> ModuleType:
    raise ImportError(f"resource module unavailable: {name}")


def _fail_journal(patch: pytest.MonkeyPatch) -> None:
    write = RunJournal._write_event

    def fail_write(journal: RunJournal, event: RunEvent) -> None:
        if event.kind == "call_completed":
            raise OSError("journal failed")
        write(journal, event)
    patch.setattr(RunJournal, "_write_event", fail_write)


def _fail_cleanup(patch: pytest.MonkeyPatch) -> None:
    release = execution.release_chats

    def fail_release(chats: tuple[LLMChat, ...], owner: object) -> None:
        release(chats, owner)
        raise RuntimeError("cleanup failed")
    patch.setattr(execution, "release_chats", fail_release)


def test_external_storage_failure_preserves_worker_outcome(tmp_path: Path) -> None:
    user = AdaptiveAttacksExperiment(LLMChat(ScriptedModel(["yes"])), Message("Base"), ContainsTextEvaluator("yes"))
    user.start()
    assert user.wait(10)
    record = capture_run(user, journal=user.journal)
    obstruction = tmp_path / "file"
    obstruction.write_text("occupied")
    with pytest.raises(OSError):
        save_run(record, obstruction)
    assert user.snapshot() == record.snapshot
    assert user.journal.error is None
    assert save_run(record, tmp_path / "retry").is_dir()


def test_stop_between_exchange_and_publication_retains_previous_evaluated_result(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = hold_boundary(monkeypatch, "call_completed", "Target", 2)
    target = LLMChat(ScriptedModel(["First", "Not accepted"]))
    user = AdaptiveAttacksExperiment(target, Message("Base"), ContainsTextEvaluator("success"))
    user.start()
    with releasing(user, release):
        assert entered.wait(5)
        user.stop()
    result, evaluation = user.result(), user.status().evaluation
    assert user.status().state is UserState.STOPPED
    assert result is not None and result.text == "First"
    assert evaluation is not None and evaluation.result_id == result.id and evaluation.success is False
    assert target.history()[-1].message.text == "Not accepted"
    assert user.status().progress.target_calls == 2 and user.status().progress.evaluator_calls == 1
