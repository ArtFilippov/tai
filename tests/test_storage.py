"""External final evidence, independent of worker ownership and later executions."""

import json
import signal
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event
from typing import overload

import pytest
from evaluation_fakes import ScriptedModel, script_success
from journal_assertions import at, mapping, released_worker, run_directory

import attacks.fitd.execution as implementation

from tai import (
    ContainsTextEvaluator,
    FITDExperiment,
    FITDUser,
    LLMChat,
    Message,
    UserState,
    capture_run,
    save_run,
)
from app.terminal import run_foreground
from core.runs.storage import RunRecord


@overload
def puzzle(user_type: type[FITDUser]) -> tuple[FITDUser, ScriptedModel, ScriptedModel]: ...


@overload
def puzzle(
    user_type: type[FITDExperiment] = FITDExperiment, *, journal_dir: Path | None = None,
) -> tuple[FITDExperiment, ScriptedModel, ScriptedModel]: ...


def puzzle(
    user_type: type[FITDUser] | type[FITDExperiment] = FITDExperiment,
    *, journal_dir: Path | None = None,
) -> tuple[FITDUser | FITDExperiment, ScriptedModel, ScriptedModel]:
    target, attacker = ScriptedModel(), ScriptedModel()
    script_success(target, attacker, "π\r\nFull answer\n")
    arguments = (
        LLMChat.cancellable(target), LLMChat.cancellable(attacker),
        Message("Fictional puzzle"), ContainsTextEvaluator("CONTROL"),
    )
    user = (
        user_type(*arguments, journal_dir=journal_dir)
        if issubclass(user_type, FITDExperiment) else user_type(*arguments)
    )
    return user, target, attacker


@pytest.mark.parametrize("user_type", [FITDUser, FITDExperiment])
def test_wait_without_storage_configuration_never_saves_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    user_type: type[FITDUser] | type[FITDExperiment],
):
    monkeypatch.chdir(tmp_path)
    user, target, attacker = puzzle(user_type)
    with pytest.raises(RuntimeError, match="finished worker"):
        capture_run(user)
    user.start()
    assert user.wait(10)
    assert user.status().state is UserState.COMPLETED
    assert not list(tmp_path.iterdir())
    settings = {"experiment": ["offline"]}
    record = capture_run(user, settings=settings)
    settings["experiment"].append("later mutation")
    assert at(record.metadata, "settings") == {"experiment": ["offline"]}
    assert record.snapshot.result is user.result()
    calls = (len(target.requests), len(attacker.requests))
    user.stop()
    assert user.wait(0)
    assert not list(tmp_path.iterdir())
    directory = save_run(record, tmp_path)
    assert (
        directory / "result.txt"
    ).read_bytes() == b"CONTROL \xcf\x80\r\nFull answer\n"
    assert json.loads((directory / "metadata.json").read_text()) == record.metadata
    assert (len(target.requests), len(attacker.requests)) == calls == (5, 9)


def test_terminal_status_cannot_be_captured_until_participants_are_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    entered, release = Event(), Event()
    release_chats = implementation.release_chats

    def hold(chats: tuple[LLMChat, ...], owner: object) -> None:
        entered.set()
        assert release.wait(5)
        release_chats(chats, owner)

    monkeypatch.setattr(implementation, "release_chats", hold)
    user, _, _ = puzzle(journal_dir=tmp_path)
    user.start()
    with released_worker(user, release):
        assert entered.wait(5)
        assert user.status().state is UserState.COMPLETED
        assert not user.wait(0)
        with pytest.raises(RuntimeError, match="finished worker"):
            capture_run(user, journal=user.journal)
        directory = run_directory(user, tmp_path)
        assert {p.name for p in directory.iterdir()} == {"events.jsonl"}
    assert {p.name for p in directory.iterdir()} == {"events.jsonl"}
    save_run(capture_run(user, journal=user.journal), tmp_path)


def assert_record_artifacts(saved: RunRecord, root: Path) -> None:
    directory = save_run(saved, root)
    assert saved.snapshot.result is not None
    assert (directory / "result.txt").read_bytes() == saved.snapshot.result.to_string().encode("utf-8")
    assert json.loads((directory / "metadata.json").read_text(encoding="utf-8")) == saved.metadata


def test_detached_record_can_be_saved_after_restart_with_original_settings(tmp_path: Path):
    user, target, attacker = puzzle()
    user.start()
    assert user.wait(10)
    first_journal = user.journal
    record = capture_run(user, journal=first_journal)
    original = record.metadata
    with pytest.raises(FrozenInstanceError):
        setattr(record.snapshot.status, "reason", "changed")
    mapping(at(record.metadata, "settings", "Target"))["model"] = "changed"
    mapping(at(first_journal.final_settings(), "Target"))["conversation_id"] = "changed"
    script_success(target, attacker, "SECOND")
    user.start()
    assert user.wait(10)
    current = capture_run(user, journal=user.journal)
    with pytest.raises(ValueError, match="Journal must belong"):
        capture_run(user, journal=first_journal)
    calls = (len(target.requests), len(attacker.requests))
    for saved in (record, current):
        assert_record_artifacts(saved, tmp_path)
    assert record.metadata == original
    assert record.snapshot.status.run_id != current.snapshot.status.run_id
    assert record.snapshot.result is not None and current.snapshot.result is not None
    assert record.snapshot.result.id != current.snapshot.result.id
    assert record.snapshot.status.evaluation is not None
    assert record.snapshot.status.evaluation.result_id == record.snapshot.result.id
    for role in ("Target", "Attacker"):
        assert (
            at(original, "final_settings", role, "conversation_id")
            != at(current.metadata, "final_settings", role, "conversation_id")
        )
    assert (len(target.requests), len(attacker.requests)) == calls == (10, 18)


@pytest.mark.parametrize("filename", ["result.txt", "metadata.json"])
def test_external_write_failure_propagates_without_changing_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str,
):
    user, _, _ = puzzle()
    user.start()
    assert user.wait(10)
    record = capture_run(user, journal=user.journal)
    events = user.journal.events()
    replace = Path.replace

    def fail(path: Path, destination: Path) -> Path:
        if destination.name == filename:
            raise OSError("output unavailable")
        return replace(path, destination)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail)
        with pytest.raises(OSError, match="output unavailable"):
            save_run(record, tmp_path)
    assert user.snapshot() == record.snapshot
    assert user.journal.error is None and user.journal.events() == events
    directory = save_run(record, tmp_path)
    assert not list(directory.glob("*.tmp"))
    assert json.loads((directory / "metadata.json").read_text())["state"] == "COMPLETED"


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("interrupt", [False, True])
def test_foreground_storage_signals_and_errors_preserve_completed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], failure: bool, interrupt: bool,
):
    user, target, attacker = puzzle(journal_dir=tmp_path)
    write = Path.write_text

    def persist(
        path: Path, data: str, encoding: str | None = None,
        errors: str | None = None, newline: str | None = None,
    ) -> int:
        if path.name == "metadata.json.tmp":
            assert user.wait(0)
            assert user.status().state is UserState.COMPLETED
            if interrupt:
                signal.raise_signal(signal.SIGINT)
                signal.raise_signal(signal.SIGINT)
            if failure:
                raise OSError("cannot save final metadata")
        return write(path, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", persist)
    handler = signal.getsignal(signal.SIGINT)
    assert run_foreground(user, artifacts_dir=tmp_path) == (
        130 if interrupt else int(failure)
    )
    assert signal.getsignal(signal.SIGINT) is handler
    assert user.status().state is UserState.COMPLETED
    assert user.journal.error is None
    assert (len(target.requests), len(attacker.requests)) == (5, 9)
    output = capsys.readouterr().out
    if failure:
        assert "Ошибка сохранения: OSError: cannot save final metadata" in output
    else:
        metadata = json.loads(
            (run_directory(user, tmp_path) / "metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["state"] == "COMPLETED"
