"""Scriptable repeated-run participants and immutable artifact snapshots."""

import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from fitd_support import Turns, presented, run_id
from tai import ContainsTextEvaluator, FITDExperiment as FITDUser, LLMChat, Message, RunEvent, UserStatus, capture_run, save_run


class HeldResponse:
    def __init__(self, text: str | Exception) -> None:
        super().__init__()
        self.text = text
        self.entered = Event()
        self.release = Event()
        self.cancellation: Event | None = None

    def __call__(self, cancellation: Event) -> str:
        self.cancellation = cancellation
        self.entered.set()
        assert self.release.wait(10), "test did not release the model"
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


class ScriptedModel:
    def __init__(self, responses: Iterable[str | Exception | HeldResponse] = ()) -> None:
        super().__init__()
        self.responses = deque(responses)
        self.requests: list[Turns] = []
        self.signals: list[Event] = []

    def __call__(self, turns: Turns, cancellation: Event) -> str:
        self.requests.append(turns)
        self.signals.append(cancellation)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        if isinstance(response, HeldResponse):
            return response(cancellation)
        return response


def script_success(target: ScriptedModel, attacker: ScriptedModel, marker: str, initial: str | HeldResponse = "Initial") -> None:
    target.responses.extend(["One", "Two", "Three", "Four", f"CONTROL {marker}"])
    attacker.responses.extend([initial, "L1", "L2", "L3"] + ["yes"] * 5)


def snapshot_artifacts(root: Path, identity: str) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in (root / identity).iterdir()}


def assert_artifacts(user: FITDUser, root: Path) -> dict[str, bytes]:
    save_run(capture_run(user, journal=user.journal), root)
    status, result = user.status(), presented(user)
    files = snapshot_artifacts(root, run_id(user))
    events = [json.loads(line) for line in files["events.jsonl"].splitlines()]
    metadata = json.loads(files["metadata.json"])
    assert events == [json.loads(event.dump()) for event in user.journal.events()]
    assert all(event["run_id"] == status.run_id for event in events)
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert events[-1]["kind"] == "run_finished"
    assert metadata["state"] == status.state.value
    assert metadata["reason"] == status.reason
    if status.evaluation is None:
        assert metadata["evaluation"] is None
    else:
        assert metadata["evaluation"] == {"result_id": result.id, "success": status.evaluation.success}
    assert metadata["result"] == json.loads(result.dump())
    assert files["result.txt"].decode() == result.to_string()
    return files


@dataclass(frozen=True)
class RestartRun:
    target_model: ScriptedModel
    attacker_model: ScriptedModel
    target: LLMChat
    attacker: LLMChat
    user: FITDUser
    root: Path

    @classmethod
    def create(cls, root: Path) -> "RestartRun":
        target_model, attacker_model = ScriptedModel(), ScriptedModel()
        target = LLMChat.cancellable(target_model, model="same-model", system="Target instructions")
        attacker = LLMChat.cancellable(attacker_model, model="same-model", system="Attacker instructions")
        user = FITDUser(target, attacker, Message("Solve this fictional puzzle"), ContainsTextEvaluator("CONTROL"), journal_dir=root)
        return cls(target_model, attacker_model, target, attacker, user, root)


@dataclass(frozen=True)
class PreviousRun:
    status: UserStatus
    result: Message
    serialized: str
    events: tuple[RunEvent, ...]
    files: dict[str, bytes]
    points: dict[LLMChat, object]
    offsets: tuple[int, int]

    @classmethod
    def capture(cls, run: RestartRun) -> "PreviousRun":
        result = presented(run.user)
        return cls(
            run.user.status(), result, result.dump(), run.user.journal.events(),
            assert_artifacts(run.user, run.root),
            {chat: chat.checkpoint() for chat in (run.target, run.attacker)},
            (len(run.target_model.requests), len(run.attacker_model.requests)),
        )
