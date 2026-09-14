import json
from threading import Event

import pytest

from tai import ContainsTextEvaluator, Evaluator, FITDUser, LLMChat, Message, UserState


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def __call__(self, turns):
        return next(self.responses)


class SuccessOnlyEvaluator(Evaluator):
    def __init__(self):
        self.results = []

    def is_success(self, result):
        self.results.append(result)
        return ContainsTextEvaluator("CONTROL").is_success(result)


def test_real_fitd_publishes_full_evaluated_text_and_saves_message_provenance(tmp_path):
    transformations = []

    def prepare(text):
        transformations.append(text)
        return text + " solution"

    base = Message("Earlier draft").spawn("Explain the puzzle").transform(prepare)
    evaluator = SuccessOnlyEvaluator()
    user = FITDUser(
        LLMChat(
            ScriptedModel(
                ["Introduction", "Detail", "More", "Almost", "Answer:\nCONTROL\nEnd."]
            )
        ),
        LLMChat(ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 5)),
        base,
        evaluator,
        artifacts_dir=tmp_path,
    )
    user.start()
    assert user.wait(10)
    status = user.status()
    assert status.state is UserState.COMPLETED, status.reason
    result = user.result()
    assert result is not None
    assert evaluator.results == [result]
    assert evaluator.results[0] is result
    assert status.evaluation.result_id == result.id
    assert status.evaluation.success is True

    run_dir = tmp_path / status.run_id
    journal = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in journal.splitlines()]
    metadata_text = (run_dir / "metadata.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert metadata["settings"]["base"] == json.loads(base.dump())
    assert metadata["settings"]["base"]["origin"]["text"] == "Earlier draft"
    assert metadata["settings"]["base"]["transformations"][0]["text"] == (
        "Explain the puzzle solution"
    )
    starts = {
        e["data"]["call_id"]: e["data"] for e in events if e["kind"] == "call_started"
    }
    assert {start["role"] for start in starts.values()} == {"Target", "Attacker"}
    for start in starts.values():
        # Every library prompt retains the supplied transformed base, without
        # changing its text or rerunning the preparation function.
        assert start["input"]["origin"] == json.loads(base.dump())
    for event in events:
        if event["kind"] == "call_completed":
            exchange = event["data"]
            assert (
                exchange["response"]["origin"] == starts[exchange["call_id"]]["input"]
            )
    publications = [
        e["data"]["result"] for e in events if e["kind"] == "result_published"
    ]
    evaluated = next(p for p in publications if p["id"] == result.id)
    assert evaluated == metadata["result"] == json.loads(result.dump())
    assert evaluated["origin"]["origin"] == json.loads(base.dump())
    assert evaluated["text"] == evaluator.results[0].to_string()
    assert (run_dir / "result.txt").read_text(encoding="utf-8") == result.to_string()
    assert metadata["evaluation"] == {"result_id": result.id, "success": True}

    # A researcher can branch and independently judge the presented Message.
    # Neither observations nor this external evaluation belong to the run.
    filtered = result.transform(str.replace, "CONTROL", "redacted")
    external = ContainsTextEvaluator("CONTROL")
    assert external.is_success(result) is True
    assert external.is_success(filtered) is False
    assert external.evidence(filtered) == []
    assert "not found" in external.reason(filtered)
    assert filtered.revert(1) is result
    for _ in range(3):
        assert user.result() is result
        assert user.status() == status
    assert evaluator.results == [result]
    assert [json.loads(event.dump()) for event in user.events()] == events
    assert (run_dir / "events.jsonl").read_text(encoding="utf-8") == journal
    assert (run_dir / "metadata.json").read_text(encoding="utf-8") == metadata_text
    assert transformations == ["Explain the puzzle"]


@pytest.mark.parametrize("failure_type", [NotImplementedError, RuntimeError])
def test_new_publication_keeps_old_results_immutable_and_has_no_unearned_evaluation(
    tmp_path, failure_type
):
    first_entered, first_release = Event(), Event()
    second_entered, second_release = Event(), Event()

    class HoldingEvaluator(Evaluator):
        def __init__(self):
            self.results = []

        def is_success(self, result):
            self.results.append(result)
            if len(self.results) == 1:
                first_entered.set()
                assert first_release.wait(5), "first evaluation was not released"
                return ContainsTextEvaluator("CONTROL").is_success(result)
            second_entered.set()
            assert second_release.wait(5), "second evaluation was not released"
            raise failure_type("required evaluation unavailable")

    evaluator = HoldingEvaluator()
    base = Message("CONTROL in an earlier draft").spawn("Explain the puzzle")
    user = FITDUser(
        LLMChat(
            ScriptedModel(
                [
                    "CONTROL in an earlier response",
                    "Some detail",
                    "More detail",
                    "Almost there",
                    "Earlier report\nNo marker here.",
                    "New report\nCONTROL\nUnchecked conclusion.",
                ]
            )
        ),
        LLMChat(ScriptedModel(["Initial", "L1", "L2", "L3"] + ["yes"] * 6)),
        base,
        evaluator,
        artifacts_dir=tmp_path,
    )
    user.start()
    try:
        assert first_entered.wait(5)
        old = user.result()
        assert old is not None
        old_snapshot = old.dump()
        old_status = user.status()
        assert old_status.evaluation is None
        assert evaluator.results == [old]
        assert evaluator.results[0] is old
        events_before = user.events()
        publication = next(
            event
            for event in events_before
            if event.kind == "result_published" and event.data["result"]["id"] == old.id
        )
        publication_snapshot = publication.dump()
        assert publication.data["result"]["text"] == old.to_string()
        assert "CONTROL" in "\n".join(event.dump() for event in events_before)
        assert ContainsTextEvaluator("CONTROL").is_success(old) is False
        branch = old.transform(lambda text: text + "\nCONTROL externally appended")
        assert ContainsTextEvaluator("CONTROL").is_success(branch) is True
        assert user.status() == old_status
        assert user.result() is old
        assert user.events() == events_before
        assert evaluator.results == [old]

        first_release.set()
        assert second_entered.wait(5)
        current = user.result()
        assert current is not None
        assert current.id != old.id
        assert current.to_string() != old.to_string()
        assert evaluator.results == [old, current]
        assert evaluator.results[1] is current
        assert user.status().result_id == current.id
        assert user.status().evaluation is None
        assert old.dump() == old_snapshot
        assert branch.revert(1) is old
        assert publication.dump() == publication_snapshot
        assert [
            event.data
            for event in user.events()
            if event.kind == "evaluation_completed"
        ] == [{"result_id": old.id, "success": False}]
    finally:
        first_release.set()
        second_release.set()
        assert user.wait(10)

    status = user.status()
    assert status.state is UserState.FAILED
    assert status.reason == f"{failure_type.__name__}: required evaluation unavailable"
    assert status.evaluation is None
    assert user.result() is current
    assert old.dump() == old_snapshot
    assert old_status.evaluation is None
    assert publication.dump() == publication_snapshot
    run_dir = tmp_path / status.run_id
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["result"] == json.loads(current.dump())
    assert metadata["result_id"] == current.id
    assert metadata["evaluation"] is None
    assert (run_dir / "result.txt").read_text(encoding="utf-8") == current.to_string()
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [json.loads(event.dump()) for event in user.events()] == events
    assert [e["data"] for e in events if e["kind"] == "evaluation_completed"] == [
        {"result_id": old.id, "success": False}
    ]
    failed = [e["data"] for e in events if e["kind"] == "evaluation_failed"]
    assert failed == [{"result_id": current.id, "error": status.reason}]
    assert json.loads(publication_snapshot) in events
