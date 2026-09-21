import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest

from fitd_support import RecordingEvaluator, ScriptedModel, evaluation, presented, releasing
from tai import (
    ContainsTextEvaluator, FITDExperiment as FITDUser, LLMChat, Message, RunEvent,
    UserState, UserStatus, capture_run, save_run,
)


def test_real_fitd_publishes_full_evaluated_text_and_saves_message_provenance(tmp_path: Path):
    transformations: list[str] = []

    def prepare(text: str) -> str:
        transformations.append(text)
        return text + " solution"

    base = Message("Earlier draft").spawn("Explain the puzzle").transform(prepare)
    evaluator = RecordingEvaluator()
    user = FITDUser(
        LLMChat(ScriptedModel(["Introduction", "Detail", "More", "Almost", "Answer:\nCONTROL\nEnd."])),
        LLMChat(ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)),
        base, evaluator, journal_dir=tmp_path,
    )
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.COMPLETED, status.reason
    result = presented(user)
    assert evaluator.results == [result]
    assert evaluator.results[0] is result
    assert evaluation(user).result_id == result.id
    assert evaluation(user).success is True
    directory = save_run(capture_run(user, journal=user.journal), tmp_path)
    _assert_saved_provenance(user, directory, evaluator)
    _assert_external_evaluation_is_independent(user, directory, evaluator)
    assert transformations == ["Explain the puzzle"]


def _assert_saved_provenance(user: FITDUser, directory: Path, evaluator: RecordingEvaluator):
    result = presented(user)
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["settings"]["base"] == json.loads(user.base.dump())
    assert metadata["settings"]["base"]["origin"]["text"] == "Earlier draft"
    assert metadata["settings"]["base"]["transformations"][0]["text"] == "Explain the puzzle solution"
    starts = {e["data"]["call_id"]: e["data"] for e in events if e["kind"] == "call_started"}
    assert {start["role"] for start in starts.values()} == {"Target", "Attacker"}
    for start in starts.values():
        assert start["input"]["origin"] == json.loads(user.base.dump())
    for event in events:
        if event["kind"] == "call_completed":
            exchange = event["data"]
            assert exchange["response"]["origin"] == starts[exchange["call_id"]]["input"]
    publications = [e["data"]["result"] for e in events if e["kind"] == "result_published"]
    evaluated = next(p for p in publications if p["id"] == result.id)
    assert evaluated == metadata["result"] == json.loads(result.dump())
    assert evaluated["origin"]["origin"] == json.loads(user.base.dump())
    assert evaluated["text"] == evaluator.results[0].to_string()
    assert (directory / "result.txt").read_text(encoding="utf-8") == result.to_string()
    assert metadata["evaluation"] == {"result_id": result.id, "success": True}


def _assert_external_evaluation_is_independent(user: FITDUser, directory: Path, evaluator: RecordingEvaluator):
    result, status = presented(user), user.status()
    journal = (directory / "events.jsonl").read_text(encoding="utf-8")
    metadata_text = (directory / "metadata.json").read_text(encoding="utf-8")
    events = [json.loads(line) for line in journal.splitlines()]
    filtered = result.transform(str.replace, "CONTROL", "redacted")
    external = ContainsTextEvaluator("CONTROL")
    assert external.evaluate(user.base, result) is True
    assert external.evaluate(user.base, filtered) is False
    assert filtered.revert(1) is result
    for _ in range(3):
        assert user.result() is result
        assert user.status() == status
    assert evaluator.results == [result]
    assert [json.loads(event.dump()) for event in user.journal.events()] == events
    assert (directory / "events.jsonl").read_text(encoding="utf-8") == journal
    assert (directory / "metadata.json").read_text(encoding="utf-8") == metadata_text


@dataclass(frozen=True)
class PublicationSnapshot:
    message: Message
    serialized: str
    status: UserStatus
    publication: RunEvent
    publication_dump: str
    branch: Message


class HoldingEvaluator:
    def __init__(self, failure_type: type[Exception]) -> None:
        super().__init__()
        self.failure_type = failure_type
        self.results: list[Message] = []
        self.first_entered, self.first_release = Event(), Event()
        self.second_entered, self.second_release = Event(), Event()

    def evaluate(self, base: Message, result: Message) -> bool:
        self.results.append(result)
        if len(self.results) == 1:
            self.first_entered.set()
            assert self.first_release.wait(5), "first evaluation was not released"
            return ContainsTextEvaluator("CONTROL").evaluate(base, result)
        self.second_entered.set()
        assert self.second_release.wait(5), "second evaluation was not released"
        raise self.failure_type("required evaluation unavailable")


@pytest.mark.parametrize("failure_type", [NotImplementedError, RuntimeError])
def test_new_publication_keeps_old_results_immutable_and_has_no_unearned_evaluation(tmp_path: Path, failure_type: type[Exception]):
    evaluator = HoldingEvaluator(failure_type)
    base = Message("CONTROL in an earlier draft").spawn("Explain the puzzle")
    user = FITDUser(
        LLMChat(ScriptedModel([
            "CONTROL in an earlier response", "Some detail", "More detail", "Almost there",
            "Earlier report\nNo marker here.", "New report\nCONTROL\nUnchecked conclusion.",
        ])),
        LLMChat(ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 6)),
        base, evaluator, journal_dir=tmp_path,
    )
    user.start()
    with releasing(user, evaluator.first_release, evaluator.second_release):
        old = _inspect_first_publication(user, evaluator)
        evaluator.first_release.set()
        current = _inspect_replacement(user, evaluator, old)
    _assert_failed_publication(user, old, current, failure_type)
    directory = save_run(capture_run(user, journal=user.journal), tmp_path)
    _assert_failed_artifacts(user, directory, old, current)


def _inspect_first_publication(user: FITDUser, evaluator: HoldingEvaluator) -> PublicationSnapshot:
    assert evaluator.first_entered.wait(5)
    old = presented(user)
    old_snapshot, old_status = old.dump(), user.status()
    assert old_status.evaluation is None
    assert evaluator.results == [old]
    assert evaluator.results[0] is old
    events_before = user.journal.events()
    publication = next(
        event for event in events_before
        if event.kind == "result_published" and json.loads(event.dump())["data"]["result"]["id"] == old.id
    )
    publication_snapshot = publication.dump()
    assert json.loads(publication.dump())["data"]["result"]["text"] == old.to_string()
    assert "CONTROL" in "\n".join(event.dump() for event in events_before)
    assert ContainsTextEvaluator("CONTROL").evaluate(user.base, old) is False
    branch = old.transform(lambda text: text + "\nCONTROL externally appended")
    assert ContainsTextEvaluator("CONTROL").evaluate(user.base, branch) is True
    assert user.status() == old_status
    assert user.result() is old
    assert user.journal.events() == events_before
    assert evaluator.results == [old]
    return PublicationSnapshot(old, old_snapshot, old_status, publication, publication_snapshot, branch)


def _inspect_replacement(user: FITDUser, evaluator: HoldingEvaluator, old: PublicationSnapshot) -> Message:
    assert evaluator.second_entered.wait(5)
    current = presented(user)
    assert current.id != old.message.id
    assert current.to_string() != old.message.to_string()
    assert evaluator.results == [old.message, current]
    assert evaluator.results[1] is current
    assert user.status().result_id == current.id
    assert user.status().evaluation is None
    assert old.message.dump() == old.serialized
    assert old.branch.revert(1) is old.message
    assert old.publication.dump() == old.publication_dump
    assert [
        {key: json.loads(event.dump())["data"][key] for key in ("result_id", "success")}
        for event in user.journal.events() if event.kind == "evaluation_completed"
    ] == [{"result_id": old.message.id, "success": False}]
    return current


def _assert_failed_publication(user: FITDUser, old: PublicationSnapshot, current: Message, failure_type: type[Exception]):
    status = user.status()
    assert status.state is UserState.FAILED
    assert status.reason == f"{failure_type.__name__}: required evaluation unavailable"
    assert status.evaluation is None
    assert user.result() is current
    assert old.message.dump() == old.serialized
    assert old.status.evaluation is None
    assert old.publication.dump() == old.publication_dump


def _assert_failed_artifacts(user: FITDUser, directory: Path, old: PublicationSnapshot, current: Message):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["result"] == json.loads(current.dump())
    assert metadata["result_id"] == current.id
    assert metadata["evaluation"] is None
    assert (directory / "result.txt").read_text(encoding="utf-8") == current.to_string()
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [json.loads(event.dump()) for event in user.journal.events()] == events
    assert [
        {key: e["data"][key] for key in ("result_id", "success")}
        for e in events if e["kind"] == "evaluation_completed"
    ] == [{"result_id": old.message.id, "success": False}]
    failed = [e["data"] for e in events if e["kind"] == "evaluation_failed"]
    assert [{key: event[key] for key in ("result_id", "error")} for event in failed] == [
        {"result_id": current.id, "error": user.status().reason},
    ]
    assert json.loads(old.publication_dump) in events
