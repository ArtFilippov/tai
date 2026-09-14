import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from tai import ContainsTextEvaluator, FITDUser, LLMChat, Message, UserState


class HeldResponse:
    def __init__(self, text):
        self.text = text
        self.entered = Event()
        self.release = Event()
        self.cancellation = None

    def __call__(self, cancellation):
        self.cancellation = cancellation
        self.entered.set()
        assert self.release.wait(10), "test did not release the model"
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


class ScriptedModel:
    def __init__(self, responses=()):
        self.responses = deque(responses)
        self.requests = []
        self.signals = []

    def __call__(self, turns, cancellation):
        self.requests.append(turns)
        self.signals.append(cancellation)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        if isinstance(response, HeldResponse):
            return response(cancellation)
        return response


def script_success(target, attacker, marker, initial="Initial"):
    target.responses.extend(["One", "Two", "Three", "Four", f"CONTROL {marker}"])
    attacker.responses.extend([initial, "L1", "L2", "L3"] + ["yes"] * 5)


def snapshot_artifacts(root, run_id):
    run_dir = root / run_id
    return {path.name: path.read_bytes() for path in run_dir.iterdir()}


def assert_artifacts(user, root):
    status, result = user.status(), user.result()
    files = snapshot_artifacts(root, status.run_id)
    events = [json.loads(line) for line in files["events.jsonl"].splitlines()]
    metadata = json.loads(files["metadata.json"])
    assert events == [json.loads(event.dump()) for event in user.events()]
    assert all(event["run_id"] == status.run_id for event in events)
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert events[-1]["kind"] == "run_finished"
    assert metadata["state"] == status.state.value
    assert metadata["reason"] == status.reason
    if status.evaluation is None:
        assert metadata["evaluation"] is None
    else:
        assert metadata["evaluation"] == {
            "result_id": result.id,
            "success": status.evaluation.success,
        }
    assert metadata["result"] == json.loads(result.dump())
    assert files["result.txt"].decode() == result.to_string()
    return files


@pytest.mark.parametrize(
    "first_outcome", ["COMPLETED", "STOPPED", "FAILED", "round_limit"]
)
def test_terminal_user_starts_an_independent_actual_fitd_run(tmp_path, first_outcome):
    target_model, attacker_model = ScriptedModel(), ScriptedModel()
    target = LLMChat.cancellable(
        target_model, model="same-model", system="Target instructions"
    )
    attacker = LLMChat.cancellable(
        attacker_model, model="same-model", system="Attacker instructions"
    )
    base = Message("Solve this fictional puzzle")
    user = FITDUser(
        target, attacker, base, ContainsTextEvaluator("CONTROL"), artifacts_dir=tmp_path
    )
    cleanup = HeldResponse("Unfinished old response")
    if first_outcome == "COMPLETED":
        script_success(target_model, attacker_model, "FIRST_ONLY")
    elif first_outcome == "round_limit":
        target_model.responses.extend(["FIRST_ONLY"] * 10)
        attacker_model.responses.extend(["Initial", "L1", "L2", "L3"] + ["yes"] * 10)
    else:
        target_model.responses.extend(
            [
                "One",
                "Two",
                "Three",
                "Four",
                "FIRST_ONLY",
                cleanup if first_outcome == "STOPPED" else RuntimeError("OLD_FAILURE"),
            ]
        )
        attacker_model.responses.extend(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)
    user.start()
    if first_outcome == "STOPPED":
        with ThreadPoolExecutor(max_workers=1) as control:
            try:
                assert cleanup.entered.wait(5)
                before = user.status()
                with pytest.raises(RuntimeError, match="active run"):
                    control.submit(user.start).result(timeout=1)
                assert user.status() == before
                assert control.submit(user.stop).result(timeout=1) is None
                assert cleanup.cancellation.wait(1)
                assert user.status().state is UserState.STOPPING
                assert not user.wait(0)
                with pytest.raises(RuntimeError, match="active run"):
                    control.submit(user.start).result(timeout=1)
                assert user.status().run_id == before.run_id
            finally:
                cleanup.release.set()
                assert user.wait(10)
    assert user.wait(10)
    old_status, old_result = user.status(), user.result()
    expected_state = (
        UserState.COMPLETED
        if first_outcome == "round_limit"
        else UserState(first_outcome)
    )
    assert old_status.state is expected_state, old_status.reason
    assert (
        old_status.reason
        == {
            "COMPLETED": "evaluator_success",
            "STOPPED": "user_stop",
            "FAILED": "RuntimeError: OLD_FAILURE",
            "round_limit": "round_limit",
        }[first_outcome]
    )
    user.stop()
    assert user.status() == old_status
    old_dump = old_result.dump()
    old_events = user.events()
    old_files = assert_artifacts(user, tmp_path)
    old_points = {chat: chat.checkpoint() for chat in (target, attacker)}
    target_offset, attacker_offset = (
        len(target_model.requests),
        len(attacker_model.requests),
    )

    held = HeldResponse("New initial request")
    target_model.responses.clear()
    attacker_model.responses.clear()
    script_success(target_model, attacker_model, "SECOND_ONLY", held)
    try:
        assert user.start() is None
        assert held.entered.wait(5)
        fresh = user.status()
        assert fresh.state is UserState.RUNNING
        assert fresh.run_id != old_status.run_id
        assert fresh.reason is None
        assert fresh.result_id is None
        assert fresh.evaluation is None
        assert fresh.progress.target_calls == fresh.progress.attacker_calls == 0
        assert user.result() is None
        assert not user.wait(0)
        assert len(target.history()) == len(attacker.history()) == 1
        clean_target = target.history()
        with pytest.raises(ValueError, match="checkpoint"):
            target.restore(old_points[target])
        assert target.history() == clean_target
        assert all(event.run_id == fresh.run_id for event in user.events())
        assert all(event not in old_events for event in user.events())
        assert held.cancellation is not attacker_model.signals[0]
        assert not held.cancellation.is_set()
    finally:
        held.release.set()
        assert user.wait(10)

    status, result = user.status(), user.result()
    assert status.state is UserState.COMPLETED, status.reason
    assert status.reason == "evaluator_success"
    assert status.progress.target_calls == 5
    assert status.progress.attacker_calls == 9
    assert status.progress.max_calls_per_conv == 10
    for chat, point in old_points.items():
        history = chat.history()
        with pytest.raises(ValueError, match="checkpoint"):
            chat.restore(point)
        assert chat.history() == history
    assert result.id != old_result.id
    assert "SECOND_ONLY" in result.to_string()
    assert "FIRST_ONLY" not in result.to_string()
    assert old_result.dump() == old_dump
    assert snapshot_artifacts(tmp_path, old_status.run_id) == old_files
    new_files = assert_artifacts(user, tmp_path)
    new_settings = json.loads(new_files["metadata.json"])["settings"]
    old_settings = json.loads(old_files["metadata.json"])["settings"]
    for role in ("Target", "Attacker"):
        assert new_settings[role].pop("conversation_id") != old_settings[role].pop(
            "conversation_id"
        )
    assert new_settings == old_settings
    assert b"FIRST_ONLY" not in b"".join(new_files.values())
    assert b"OLD_FAILURE" not in b"".join(new_files.values())
    for model, offset, system in [
        (target_model, target_offset, "Target instructions"),
        (attacker_model, attacker_offset, "Attacker instructions"),
    ]:
        first = model.requests[offset]
        assert [turn.role for turn in first] == ["system", "user"]
        assert first[0].message.to_string() == system
    assert (
        base.to_string()
        in attacker_model.requests[attacker_offset][-1].message.to_string()
    )
    assert target.model == attacker.model == "same-model"
    user.stop()
    assert user.status() == status
    assert assert_artifacts(user, tmp_path) == new_files


@pytest.mark.parametrize("shared_role", ["Target", "Attacker"])
@pytest.mark.parametrize(
    "terminal", [UserState.COMPLETED, UserState.STOPPED, UserState.FAILED]
)
def test_chat_ownership_conflict_is_atomic_and_released_after_terminal_state(
    tmp_path, shared_role, terminal
):
    held = HeldResponse(
        RuntimeError("Owner failed") if terminal is UserState.FAILED else "Two"
    )
    target_model = ScriptedModel(["One", held, "Three", "Four", "CONTROL owner"])
    attacker_model = ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)
    target = LLMChat.cancellable(target_model)
    attacker = LLMChat.cancellable(attacker_model)
    owner = FITDUser(
        target,
        attacker,
        Message("Owner task"),
        ContainsTextEvaluator("CONTROL"),
        artifacts_dir=tmp_path,
    )
    shared_chat, shared_model = (
        (target, target_model)
        if shared_role == "Target"
        else (attacker, attacker_model)
    )
    free_model = ScriptedModel(["Existing independent conversation"])
    free_chat = LLMChat.cancellable(free_model)
    free_chat.send(Message("Earlier independent request"))
    contender = FITDUser(
        free_chat,
        shared_chat,
        Message("Contender task"),
        ContainsTextEvaluator("CONTROL"),
        artifacts_dir=tmp_path,
    )
    owner.start()
    try:
        assert held.entered.wait(5)
        owner_status, owner_events, owner_result = (
            owner.status(),
            owner.events(),
            owner.result(),
        )
        histories = [chat.history() for chat in (target, attacker, free_chat)]
        with pytest.raises(RuntimeError, match="LLMChat.*active"):
            contender.start()
        assert contender.status().state is UserState.IDLE
        assert contender.status().run_id is None
        assert contender.events() == ()
        assert contender.result() is None
        assert owner.status() == owner_status
        assert owner.events() == owner_events
        assert owner.result() is owner_result
        assert [chat.history() for chat in (target, attacker, free_chat)] == histories

        # The rejected pair must not reserve its free participant.
        independent_attacker = ScriptedModel()
        script_success(free_model, independent_attacker, "independent")
        independent = FITDUser(
            free_chat,
            LLMChat.cancellable(independent_attacker),
            Message("Another task"),
            ContainsTextEvaluator("CONTROL"),
            artifacts_dir=tmp_path,
        )
        independent.start()
        assert independent.wait(10)
        assert independent.status().state is UserState.COMPLETED, (
            independent.status().reason
        )
    finally:
        if terminal is UserState.STOPPED:
            owner.stop()
        held.release.set()
        assert owner.wait(10)
        contender.stop()
        contender.wait(10)
    assert owner.status().state is terminal, owner.status().reason
    owner_files = assert_artifacts(owner, tmp_path)

    shared_model.responses.clear()
    script_success(free_model, shared_model, "contender")
    contender.start()
    assert contender.wait(10)
    assert contender.status().state is UserState.COMPLETED, contender.status().reason
    assert "contender" in contender.result().to_string()
    contender_files = assert_artifacts(contender, tmp_path)
    assert b"Owner task" not in b"".join(contender_files.values())
    assert b"CONTROL owner" not in b"".join(contender_files.values())
    assert snapshot_artifacts(tmp_path, owner.status().run_id) == owner_files


def test_simultaneous_users_acquire_opposite_role_pairs_without_partial_ownership(
    tmp_path,
):
    entered, release, go = Event(), Event(), Event()
    ready = [Event(), Event()]

    def held_model(turns, cancellation):
        entered.set()
        assert release.wait(10), "test did not release the model"
        return "Initial"

    chats = [LLMChat.cancellable(held_model), LLMChat.cancellable(held_model)]
    users = [
        FITDUser(
            chats[index],
            chats[1 - index],
            Message(f"Task {index}"),
            ContainsTextEvaluator("CONTROL"),
            artifacts_dir=tmp_path,
        )
        for index in range(2)
    ]

    def start(index):
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
            assert loser.events() == ()
            assert all(chat.history() == () for chat in chats)
            assert len(list(tmp_path.iterdir())) == 1
            winner.stop()
            assert winner.status().state is UserState.STOPPING
            with pytest.raises(RuntimeError, match="LLMChat.*active"):
                loser.start()
        finally:
            go.set()
            for user in users:
                user.stop()
            release.set()
            for user in users:
                user.wait(10)
    assert winner.status().state is UserState.STOPPED


def test_failed_artifact_setup_releases_chats_for_another_actual_run(
    tmp_path, monkeypatch
):
    target_model, attacker_model = ScriptedModel(), ScriptedModel()
    target, attacker = (
        LLMChat.cancellable(target_model),
        LLMChat.cancellable(attacker_model),
    )
    owner = FITDUser(
        target,
        attacker,
        Message("Task"),
        ContainsTextEvaluator("CONTROL"),
        artifacts_dir=tmp_path,
    )
    mkdir = Path.mkdir

    def unavailable(path, *args, **kwargs):
        if path.parent == tmp_path:
            raise OSError("artifact storage unavailable")
        return mkdir(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", unavailable)
        owner.start()
        assert owner.wait(10)
    status = owner.status()
    assert status.state is UserState.FAILED
    assert "artifact storage unavailable" in status.reason
    assert owner.result() is None
    assert target.history() == attacker.history() == ()
    script_success(target_model, attacker_model, "recovered")
    owner.start()
    assert owner.wait(10)
    assert owner.status().state is UserState.COMPLETED, owner.status().reason
    assert owner.status().run_id != status.run_id
    assert_artifacts(owner, tmp_path)
