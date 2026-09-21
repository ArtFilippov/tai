import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from fitd_restart_support import (
    HeldResponse, PreviousRun, RestartRun, ScriptedModel, assert_artifacts, script_success, snapshot_artifacts,
)
from fitd_support import Turns, presented, reason, releasing, run_id
from tai import ContainsTextEvaluator, FITDExperiment as FITDUser, LLMChat, Message, UserState


@pytest.mark.parametrize("first_outcome", ["COMPLETED", "STOPPED", "FAILED", "round_limit"])
def test_terminal_user_starts_an_independent_actual_fitd_run(tmp_path: Path, first_outcome: str):
    run = RestartRun.create(tmp_path)
    _finish_first_run(run, first_outcome)
    old = PreviousRun.capture(run)
    held = HeldResponse("New initial request")
    run.target_model.responses.clear()
    run.attacker_model.responses.clear()
    script_success(run.target_model, run.attacker_model, "SECOND_ONLY", held)
    with releasing(run.user, held.release):
        assert run.user.start() is None
        assert held.entered.wait(5)
        _assert_fresh_state(run, old, held)
    _assert_second_run(run, old)


def _finish_first_run(run: RestartRun, first_outcome: str):
    cleanup = HeldResponse("Unfinished old response")
    if first_outcome == "COMPLETED":
        script_success(run.target_model, run.attacker_model, "FIRST_ONLY")
    elif first_outcome == "round_limit":
        run.target_model.responses.extend(["FIRST_ONLY"] * 10)
        run.attacker_model.responses.extend(["Initial", "L1", "L2", "L3"] + ["yes"] * 10)
    else:
        run.target_model.responses.extend([
            "One", "Two", "Three", "Four", "FIRST_ONLY",
            cleanup if first_outcome == "STOPPED" else RuntimeError("OLD_FAILURE"),
        ])
        run.attacker_model.responses.extend(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)
    run.user.start()
    if first_outcome == "STOPPED":
        with ThreadPoolExecutor(max_workers=1) as control, releasing(run.user, cleanup.release):
            _assert_held_stop(run.user, cleanup, control)
    assert run.user.wait(10)
    status = run.user.status()
    expected = UserState.COMPLETED if first_outcome == "round_limit" else UserState(first_outcome)
    assert status.state is expected, status.reason
    assert status.reason == {
        "COMPLETED": "evaluator_success", "STOPPED": "user_stop",
        "FAILED": "RuntimeError: OLD_FAILURE", "round_limit": "round_limit",
    }[first_outcome]
    run.user.stop()
    assert run.user.status() == status


def _assert_held_stop(user: FITDUser, cleanup: HeldResponse, control: ThreadPoolExecutor):
    assert cleanup.entered.wait(5)
    before = user.status()
    with pytest.raises(RuntimeError, match="active run"):
        control.submit(user.start).result(timeout=1)
    assert user.status() == before
    assert control.submit(user.stop).result(timeout=1) is None
    assert cleanup.cancellation is not None
    assert not cleanup.cancellation.is_set()
    assert user.status().state is UserState.STOPPING
    assert not user.wait(0)
    with pytest.raises(RuntimeError, match="active run"):
        control.submit(user.start).result(timeout=1)
    assert user.status().run_id == before.run_id


def _assert_fresh_state(run: RestartRun, old: PreviousRun, held: HeldResponse):
    user = run.user
    fresh = user.status()
    assert fresh.state is UserState.RUNNING
    assert fresh.run_id != old.status.run_id
    assert fresh.reason is None
    assert fresh.result_id is None
    assert fresh.evaluation is None
    assert fresh.progress.target_calls == fresh.progress.attacker_calls == 0
    assert user.result() is None
    assert not user.wait(0)
    assert len(run.target.history()) == len(run.attacker.history()) == 1
    clean_target = run.target.history()
    with pytest.raises(ValueError, match="checkpoint"):
        run.target.restore(old.points[run.target])
    assert run.target.history() == clean_target
    assert all(event.run_id == fresh.run_id for event in user.journal.events())
    assert all(event not in old.events for event in user.journal.events())
    assert held.cancellation is not run.attacker_model.signals[0]
    assert held.cancellation is not None
    assert not held.cancellation.is_set()


def _assert_second_run(run: RestartRun, old: PreviousRun):
    user = run.user
    status, result = user.status(), presented(user)
    assert status.state is UserState.COMPLETED, status.reason
    assert status.reason == "evaluator_success"
    assert status.progress.target_calls == 5
    assert status.progress.attacker_calls == 9
    assert status.progress.max_calls_per_conv == 10
    for chat, point in old.points.items():
        history = chat.history()
        with pytest.raises(ValueError, match="checkpoint"):
            chat.restore(point)
        assert chat.history() == history
    assert result.id != old.result.id
    assert "SECOND_ONLY" in result.to_string()
    assert "FIRST_ONLY" not in result.to_string()
    assert old.result.dump() == old.serialized
    assert old.status.run_id is not None
    assert snapshot_artifacts(run.root, old.status.run_id) == old.files
    new_files = assert_artifacts(user, run.root)
    _assert_fresh_artifacts(run, old, new_files)
    user.stop()
    assert user.status() == status
    assert assert_artifacts(user, run.root) == new_files


def _assert_fresh_artifacts(run: RestartRun, old: PreviousRun, new_files: dict[str, bytes]):
    new_settings = json.loads(new_files["metadata.json"])["settings"]
    old_settings = json.loads(old.files["metadata.json"])["settings"]
    for role in ("Target", "Attacker"):
        assert new_settings[role].pop("conversation_id") != old_settings[role].pop("conversation_id")
    assert new_settings == old_settings
    assert b"FIRST_ONLY" not in b"".join(new_files.values())
    assert b"OLD_FAILURE" not in b"".join(new_files.values())
    target_offset, attacker_offset = old.offsets
    for model, offset, system in (
        (run.target_model, target_offset, "Target instructions"),
        (run.attacker_model, attacker_offset, "Attacker instructions"),
    ):
        first = model.requests[offset]
        assert [turn.role for turn in first] == ["system", "user"]
        assert first[0].message.to_string() == system
    assert run.user.base.to_string() in run.attacker_model.requests[attacker_offset][-1].message.to_string()
    assert run.target.model == run.attacker.model == "same-model"


@pytest.mark.parametrize("shared_role", ["Target", "Attacker"])
@pytest.mark.parametrize("terminal", [UserState.COMPLETED, UserState.STOPPED, UserState.FAILED])
def test_chat_ownership_conflict_is_atomic_and_released_after_terminal_state(
    tmp_path: Path, shared_role: str, terminal: UserState,
):
    held = HeldResponse(RuntimeError("Owner failed") if terminal is UserState.FAILED else "Two")
    target_model = ScriptedModel(["One", held, "Three", "Four", "CONTROL owner"])
    attacker_model = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)
    target, attacker = LLMChat.cancellable(target_model), LLMChat.cancellable(attacker_model)
    owner = FITDUser(target, attacker, Message("Owner task"), ContainsTextEvaluator("CONTROL"), journal_dir=tmp_path)
    shared_chat, shared_model = (target, target_model) if shared_role == "Target" else (attacker, attacker_model)
    free_model = ScriptedModel(["Existing independent conversation"])
    free_chat = LLMChat.cancellable(free_model)
    free_chat.send(Message("Earlier independent request"))
    contender = FITDUser(free_chat, shared_chat, Message("Contender task"), ContainsTextEvaluator("CONTROL"), journal_dir=tmp_path)
    owner.start()
    try:
        _assert_ownership_conflict(owner, contender, held, (target, attacker, free_chat))
        _run_independent(tmp_path, free_chat, free_model)
    finally:
        if terminal is UserState.STOPPED:
            owner.stop()
        held.release.set()
        assert owner.wait(10)
        contender.stop()
        contender.wait(10)
    assert owner.status().state is terminal, owner.status().reason
    owner_files = assert_artifacts(owner, tmp_path)
    _run_contender(tmp_path, contender, free_model, shared_model)
    assert snapshot_artifacts(tmp_path, run_id(owner)) == owner_files


def _assert_ownership_conflict(owner: FITDUser, contender: FITDUser, held: HeldResponse, chats: tuple[LLMChat, ...]):
    assert held.entered.wait(5)
    owner_status, owner_events, owner_result = owner.status(), owner.journal.events(), owner.result()
    histories = [chat.history() for chat in chats]
    with pytest.raises(RuntimeError, match="LLMChat.*active"):
        contender.start()
    assert contender.status().state is UserState.IDLE
    assert contender.status().run_id is None
    assert contender.journal.events() == ()
    assert contender.result() is None
    assert owner.status() == owner_status
    assert owner.journal.events() == owner_events
    assert owner.result() is owner_result
    assert [chat.history() for chat in chats] == histories


def _run_independent(root: Path, free_chat: LLMChat, free_model: ScriptedModel):
    independent_attacker = ScriptedModel()
    script_success(free_model, independent_attacker, "independent")
    independent = FITDUser(
        free_chat, LLMChat.cancellable(independent_attacker), Message("Another task"),
        ContainsTextEvaluator("CONTROL"), journal_dir=root,
    )
    independent.start()
    assert independent.wait(10)
    assert independent.status().state is UserState.COMPLETED, independent.status().reason


def _run_contender(root: Path, contender: FITDUser, free_model: ScriptedModel, shared_model: ScriptedModel):
    shared_model.responses.clear()
    script_success(free_model, shared_model, "contender")
    contender.start()
    assert contender.wait(10)
    assert contender.status().state is UserState.COMPLETED, contender.status().reason
    assert "contender" in presented(contender).to_string()
    contender_files = assert_artifacts(contender, root)
    assert b"Owner task" not in b"".join(contender_files.values())
    assert b"CONTROL owner" not in b"".join(contender_files.values())


def test_simultaneous_users_acquire_opposite_role_pairs_without_partial_ownership(tmp_path: Path):
    entered, release, go = Event(), Event(), Event()
    ready = [Event(), Event()]

    def held_model(turns: Turns, cancellation: Event) -> str:
        entered.set()
        assert release.wait(10), "test did not release the model"
        return "Initial"

    chats = [LLMChat.cancellable(held_model), LLMChat.cancellable(held_model)]
    users = [
        FITDUser(chats[index], chats[1 - index], Message(f"Task {index}"), ContainsTextEvaluator("CONTROL"), journal_dir=tmp_path)
        for index in range(2)
    ]

    def start(index: int) -> str | None:
        ready[index].set()
        assert go.wait(5)
        try:
            users[index].start()
        except RuntimeError as exc:
            return str(exc)
        return None

    with ThreadPoolExecutor(max_workers=2) as control:
        pending = [control.submit(start, index) for index in range(2)]
        try:
            winner = _assert_race(tmp_path, users, chats, pending, (entered, go, *ready))
        finally:
            go.set()
            for user in users:
                user.stop()
            release.set()
            for user in users:
                user.wait(10)
    assert winner.status().state is UserState.STOPPED


def _assert_race(
    root: Path, users: list[FITDUser], chats: list[LLMChat],
    pending: list[Future[str | None]], signals: tuple[Event, ...],
) -> FITDUser:
    entered, go, *ready = signals
    assert all(signal.wait(5) for signal in ready)
    go.set()
    outcomes = [future.result(timeout=1) for future in pending]
    assert outcomes.count(None) == 1
    rejected = next(outcome for outcome in outcomes if outcome is not None)
    assert "LLMChat" in rejected and "active" in rejected
    winner_index = outcomes.index(None)
    winner, loser = users[winner_index], users[1 - winner_index]
    assert entered.wait(5)
    assert winner.status().state is UserState.RUNNING
    assert loser.status().state is UserState.IDLE
    assert loser.journal.events() == ()
    assert all(chat.history() == () for chat in chats)
    assert len(list(root.iterdir())) == 1
    winner.stop()
    assert winner.status().state is UserState.STOPPING
    with pytest.raises(RuntimeError, match="LLMChat.*active"):
        loser.start()
    return winner


def test_failed_journal_setup_releases_chats_for_another_actual_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target_model, attacker_model = ScriptedModel(), ScriptedModel()
    target, attacker = LLMChat.cancellable(target_model), LLMChat.cancellable(attacker_model)
    owner = FITDUser(target, attacker, Message("Task"), ContainsTextEvaluator("CONTROL"), journal_dir=tmp_path)
    mkdir = Path.mkdir

    def unavailable(path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        if path.parent == tmp_path:
            raise OSError("artifact storage unavailable")
        mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", unavailable)
        owner.start()
        assert owner.wait(10)
    status = owner.status()
    assert status.state is UserState.FAILED
    assert "artifact storage unavailable" in reason(owner)
    assert owner.result() is None
    assert target.history() == attacker.history() == ()
    script_success(target_model, attacker_model, "recovered")
    owner.start()
    assert owner.wait(10)
    assert owner.status().state is UserState.COMPLETED, owner.status().reason
    assert owner.status().run_id != status.run_id
    assert_artifacts(owner, tmp_path)
