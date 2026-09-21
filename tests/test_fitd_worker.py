"""Worker failures and cleanup are observable independently of User.wait()."""

from pathlib import Path
from threading import Event

import pytest
from fitd_observation import Observations, install_observers, puzzle
from fitd_support import ScriptedModel, Turns

import attacks.fitd.execution as implementation
from attacks.fitd.native import FITD, GarakException
from tai import ContainsTextEvaluator, FITDExperiment as FITDUser, LLMChat, Message, RunEvent, RunJournal, UserState


@pytest.fixture
def observed(monkeypatch: pytest.MonkeyPatch) -> Observations:
    return install_observers(monkeypatch)


def _inject_failure(monkeypatch: pytest.MonkeyPatch, failure: str, root: Path, error: Exception) -> None:
    if failure == "decision":
        def fail(*args: object) -> bool:
            raise error
        monkeypatch.setattr(FITD, "_is_refusal", fail)
    elif failure == "journal":
        _fail_journal(monkeypatch, error)
    elif failure == "setup":
        _fail_setup(monkeypatch, root, error)


def _fail_journal(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    write = RunJournal._write_event

    def fail_journal(journal: RunJournal, event: RunEvent) -> None:
        if event.kind == "run_finished":
            raise error
        write(journal, event)
    monkeypatch.setattr(RunJournal, "_write_event", fail_journal)


def _fail_setup(monkeypatch: pytest.MonkeyPatch, root: Path, error: Exception) -> None:
    mkdir = Path.mkdir

    def fail_directory(path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        if path.parent == root:
            raise error
        mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)
    monkeypatch.setattr(Path, "mkdir", fail_directory)


@pytest.mark.parametrize("failure", ["provider", "decision", "journal", "setup"])
def test_worker_returns_actual_failures_including_errors_caught_by_garak(
    tmp_path: Path, observed: Observations, monkeypatch: pytest.MonkeyPatch, failure: str,
):
    error = GarakException("native boundary failed")
    responses: list[str | Exception] = [error] if failure == "provider" else ["Opening"]
    _inject_failure(monkeypatch, failure, tmp_path, error)
    user, target, attacker = puzzle(tmp_path, responses, ["Initial", "L1", "L2", "L3", "yes"], 1)
    user.start()
    assert observed.finished.wait(10)
    outcome = observed.values["_run"][0]
    assert outcome.status.state is UserState.FAILED
    assert outcome.status.reason == "GarakException: native boundary failed"
    assert outcome.error is error
    assert outcome.released is not None
    assert outcome.released.roles == ("Target", "Attacker")
    next_user = FITDUser(
        target, attacker, Message("Another puzzle"), ContainsTextEvaluator("CONTROL"),
        journal_dir=tmp_path, max_calls_per_conv=1,
    )
    next_user.start()
    assert next_user.wait(10)


@pytest.mark.parametrize("stop", [False, True])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_start_return_is_distinct_from_work_and_resource_cleanup(
    tmp_path: Path, observed: Observations, monkeypatch: pytest.MonkeyPatch, stop: bool, cleanup_failure: bool,
):
    entered, allow_response = Event(), Event()
    releasing, allow_release = Event(), Event()
    release_chats = implementation.release_chats

    def target_model(turns: Turns) -> str:
        entered.set()
        assert allow_response.wait(5)
        return "Opening"

    def cleanup(chats: tuple[LLMChat, ...], owner: object) -> None:
        releasing.set()
        assert allow_release.wait(5)
        release_chats(chats, owner)
        if cleanup_failure:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(implementation, "release_chats", cleanup)
    user = FITDUser(
        LLMChat(target_model), LLMChat(ScriptedModel(["Initial", "L1", "L2", "L3", "yes"])),
        Message("Puzzle"), ContainsTextEvaluator("CONTROL"), journal_dir=tmp_path, max_calls_per_conv=1,
    )
    try:
        _assert_cleanup_in_progress(user, observed, (entered, allow_response, releasing), stop)
    finally:
        allow_response.set()
        allow_release.set()
    assert observed.finished.wait(10)
    _assert_cleanup_outcome(observed, stop, cleanup_failure)


def _assert_cleanup_in_progress(user: FITDUser, observed: Observations, signals: tuple[Event, Event, Event], stop: bool):
    entered, allow_response, releasing = signals
    assert user.start() is None
    assert entered.wait(5)
    assert observed.values["start"][0].run_id == user.status().run_id
    assert not observed.finished.is_set()
    if stop:
        user.stop()
    allow_response.set()
    assert releasing.wait(5)
    assert user.status().state is (UserState.STOPPED if stop else UserState.COMPLETED)
    assert not observed.finished.is_set()
    assert not observed.values["_release_resources"]
    terminal = user.status()
    user.stop()
    assert user.status() == terminal
    assert not user.wait(0)
    with pytest.raises(RuntimeError, match="active run"):
        user.start()


def _assert_cleanup_outcome(observed: Observations, stop: bool, cleanup_failure: bool):
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
        assert observed.values["exchange"][-1].exchange.response.to_string() == "Opening"
