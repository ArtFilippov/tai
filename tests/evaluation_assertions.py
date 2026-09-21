"""Correlations between evaluator requests, verdicts, and saved run evidence."""

import json
from collections.abc import Sequence

from journal_assertions import at, mapping, text

from core.chats import ChatTurn
from core.evaluation.model_evaluator import MODEL_RUBRIC
from core.primitives.messages import Message


def classification_data(turns: Sequence[ChatTurn]) -> dict[str, object]:
    rubric, data = turns[-1].message.to_string().split("\n\nINPUT_JSON:\n", 1)
    assert rubric == MODEL_RUBRIC
    return mapping(json.loads(data))


def event_data(
    events: list[dict[str, object]], kind: str, role: str | None = None
) -> list[dict[str, object]]:
    return [
        mapping(event["data"]) for event in events
        if event["kind"] == kind and (role is None or at(event, "data", "role") == role)
    ]


def assert_linked_exchange(
    base: Message, evidence: tuple[dict[str, object], ...],
    published: dict[str, dict[str, object]], turns: Sequence[ChatTurn],
) -> None:
    assessment, start, end, evaluation = evidence
    assert assessment["base_id"] == start["base_id"] == evaluation["base_id"] == base.id
    assert assessment["base"] == json.loads(base.dump())
    assert assessment["result"] == published[text(start["result_id"])]
    assert (
        assessment["evaluation_id"] == start["evaluation_id"]
        == end["evaluation_id"] == evaluation["evaluation_id"]
    )
    assert start["call_id"] == end["call_id"]
    assert start["result_id"] == end["result_id"] == evaluation["result_id"]
    assert at(start, "input", "id") == at(end, "response", "origin", "id")
    assert classification_data(turns) == {
        "base_request": base.to_string(),
        "result": published[text(start["result_id"])]["text"],
    }
    assert json.loads(text(at(end, "response", "text")))["success"] is evaluation["success"]


def assert_assessment_events(
    events: list[dict[str, object]], base: Message, verdicts: list[bool],
    requests: list[tuple[ChatTurn, ...]],
) -> None:
    starts = event_data(events, "call_started", "Evaluator")
    ends = event_data(events, "call_completed", "Evaluator")
    evaluations = event_data(events, "evaluation_completed")
    assessments = event_data(events, "evaluation_started")
    assert len({text(e["evaluation_id"]) for e in assessments}) == len(verdicts)
    assert [e["success"] for e in evaluations] == verdicts
    published = {
        text(at(event, "result", "id")): mapping(event["result"])
        for event in event_data(events, "result_published")
    }
    assert len({text(start["conversation_id"]) for start in starts}) == len(verdicts)
    assert all(len(turns) == 2 for turns in requests)
    assert not {text(start["conversation_id"]) for start in starts} & {
        text(event["conversation_id"]) for event in event_data(events, "call_started")
        if event["role"] != "Evaluator"
    }
    for assessment, start, end, evaluation, turns in zip(
        assessments, starts, ends, evaluations, requests, strict=True
    ):
        assert_linked_exchange(base, (assessment, start, end, evaluation), published, turns)
