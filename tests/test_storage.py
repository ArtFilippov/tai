"""External final evidence, independent of worker ownership and later executions."""

import json
import signal
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event

import pytest
from test_fitd_restart import ScriptedModel, script_success

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
from tai.app.terminal import run_foreground


def puzzle(user_type=FITDExperiment, **kwargs):
    target, attacker = ScriptedModel(), ScriptedModel()
    script_success(target, attacker, "π\r\nFull answer\n")
    user = user_type(
        LLMChat.cancellable(target),
        LLMChat.cancellable(attacker),
        Message("Fictional puzzle"),
        ContainsTextEvaluator("CONTROL"),
        **kwargs,
    )
    return user, target, attacker


@pytest.mark.parametrize("user_type", [FITDUser, FITDExperiment])
def test_wait_without_storage_configuration_never_saves_results(
    tmp_path, monkeypatch, user_type
):
    monkeypatch.chdir(tmp_path)
    user, target, attacker = puzzle(user_type)
    with pytest.raises(RuntimeError, match="finished worker"):
        capture_run(user)
    user.start()
    assert user.wait(10)
    assert user.status().state is UserState.COMPLETED
    assert list(tmp_path.iterdir()) == []
    settings = {"experiment": ["offline"]}
    record = capture_run(user, settings=settings)
    settings["experiment"].append("later mutation")
    assert record.metadata["settings"] == {"experiment": ["offline"]}
    assert record.snapshot.result is user.result()
    calls = (len(target.requests), len(attacker.requests))
    user.stop()
    assert user.wait(0)
    assert list(tmp_path.iterdir()) == []
    directory = save_run(record, tmp_path)
    assert (
        directory / "result.txt"
    ).read_bytes() == b"CONTROL \xcf\x80\r\nFull answer\n"
    assert json.loads((directory / "metadata.json").read_text()) == record.metadata
    assert (len(target.requests), len(attacker.requests)) == calls == (5, 9)


def test_terminal_status_cannot_be_captured_until_participants_are_released(
    tmp_path, monkeypatch
):
    import tai.attacks.fitd.user as implementation

    entered, release = Event(), Event()
    release_chats = implementation._release_chats

    def hold(chats, owner):
        entered.set()
        assert release.wait(5)
        release_chats(chats, owner)

    monkeypatch.setattr(implementation, "_release_chats", hold)
    user, _, _ = puzzle(journal_dir=tmp_path)
    user.start()
    try:
        assert entered.wait(5)
        assert user.status().state is UserState.COMPLETED
        assert not user.wait(0)
        with pytest.raises(RuntimeError, match="finished worker"):
            capture_run(user, journal=user.journal)
        directory = tmp_path / user.status().run_id
        assert {p.name for p in directory.iterdir()} == {"events.jsonl"}
    finally:
        release.set()
        assert user.wait(10)
    assert {p.name for p in directory.iterdir()} == {"events.jsonl"}
    save_run(capture_run(user, journal=user.journal), tmp_path)


def test_detached_record_can_be_saved_after_restart_with_original_settings(tmp_path):
    user, target, attacker = puzzle()
    user.start()
    assert user.wait(10)
    first_journal = user.journal
    record = capture_run(user, journal=first_journal)
    original = record.metadata
    with pytest.raises(FrozenInstanceError):
        record.snapshot.status.reason = "changed"
    record.metadata["settings"]["Target"]["model"] = "changed"
    first_journal.final_settings()["Target"]["conversation_id"] = "changed"
    script_success(target, attacker, "SECOND")
    user.start()
    assert user.wait(10)
    current = capture_run(user, journal=user.journal)
    with pytest.raises(ValueError, match="Journal must belong"):
        capture_run(user, journal=first_journal)
    calls = (len(target.requests), len(attacker.requests))
    for saved in (record, current):
        directory = save_run(saved, tmp_path)
        assert (directory / "result.txt").read_bytes() == (
            saved.snapshot.result.to_string().encode("utf-8")
        )
        assert json.loads((directory / "metadata.json").read_text()) == saved.metadata
    assert record.metadata == original
    assert record.snapshot.status.run_id != current.snapshot.status.run_id
    assert record.snapshot.result.id != current.snapshot.result.id
    assert record.snapshot.status.evaluation.result_id == record.snapshot.result.id
    for role in ("Target", "Attacker"):
        assert (
            original["final_settings"][role]["conversation_id"]
            != current.metadata["final_settings"][role]["conversation_id"]
        )
    assert (len(target.requests), len(attacker.requests)) == calls == (10, 18)


@pytest.mark.parametrize("filename", ["result.txt", "metadata.json"])
def test_external_write_failure_propagates_without_changing_user(
    tmp_path, monkeypatch, filename
):
    user, _, _ = puzzle()
    user.start()
    assert user.wait(10)
    record = capture_run(user, journal=user.journal)
    events = user.journal.events()
    replace = Path.replace

    def fail(path, destination):
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
    tmp_path, monkeypatch, capsys, failure, interrupt
):
    user, target, attacker = puzzle(journal_dir=tmp_path)
    write = Path.write_text

    def persist(path, *args, **kwargs):
        if path.name == "metadata.json.tmp":
            assert user.wait(0)
            assert user.status().state is UserState.COMPLETED
            if interrupt:
                signal.raise_signal(signal.SIGINT)
                signal.raise_signal(signal.SIGINT)
            if failure:
                raise OSError("cannot save final metadata")
        return write(path, *args, **kwargs)

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
            (tmp_path / user.status().run_id / "metadata.json").read_text()
        )
        assert metadata["state"] == "COMPLETED"
