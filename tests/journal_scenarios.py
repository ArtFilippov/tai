"""Typed participants and evidence checks for independent journal scenarios."""

from collections import Counter
from collections.abc import Callable, Iterable

import pytest
from journal_assertions import at, mapping, sequence, text

from tai import ChatTurn, EvaluatorSetup, LLMChat, Message
from attacks.fitd.native import FITD
from core.runs.journal import RunEvent

type Trace = list[tuple[str, tuple[tuple[str, str], ...]]]


def participants(
    targets: Iterable[str], continuations: Iterable[str],
    verdicts: Iterable[str] = (' {"success": true}',),
) -> tuple[LLMChat, LLMChat, Message, EvaluatorSetup, Trace]:
    trace: Trace = []

    def chat(role: str, replies: Iterable[str]) -> LLMChat:
        responses = iter(replies)

        def generate(turns: tuple[ChatTurn, ...]) -> str:
            trace.append((role, tuple((turn.role, turn.message.to_string()) for turn in turns)))
            return next(responses)

        return LLMChat(generate, system=role + " instructions")

    target = chat("Target", targets)
    attacker = chat("Attacker", ["Initial", "L1", "L2", "L3", *continuations])
    judge = chat("Evaluator", verdicts)
    return target, attacker, Message("Solve the fictional puzzle"), EvaluatorSetup.model(judge), trace


def count_native_checks(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    counts: Counter[str] = Counter()

    def count(name: str) -> None:
        operation: Callable[..., object] = getattr(FITD, name)

        def invoke(*args: object, **kwargs: object) -> object:
            counts[name] += 1
            return operation(*args, **kwargs)

        monkeypatch.setattr(FITD, name, invoke)

    for name in (
        "_is_refusal", "_is_response_aligned", "_is_realign_prompt", "generate_next_turn_attempt",
    ):
        count(name)
    return counts


def assert_initialization(events: tuple[RunEvent, ...], run_id: str | None) -> None:
    assert events[0].kind == "run_started"
    assert events[-2].kind == "resources_released"
    assert events[-1].kind == "run_finished"
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert all(event.run_id == run_id for event in events)
    resets = [
        mapping(event.data) for event in events
        if event.kind == "history_reset"
        and at(event.data, "context", "cause", "reason") == "run_started"
    ]
    assert [event["role"] for event in resets] == ["Target", "Attacker", "Evaluator"]
    initial = next(event for event in events if event.kind == "checkpoint_created")
    assert at(initial.data, "context", "cause", "reason") == "initial_checkpoint"
    assert initial.sequence < next(event.sequence for event in events if event.kind == "call_started")


def keyed_events(
    events: tuple[RunEvent, ...], kind: str, key: str,
) -> dict[str, dict[str, object]]:
    return {
        text(at(event.data, key)): mapping(event.data) for event in events if event.kind == kind
    }


def assert_branch_links(
    events: tuple[RunEvent, ...], ends: dict[str, dict[str, object]],
) -> None:
    exchanges = keyed_events(events, "target_exchange", "checkpoint_id")
    branches = [mapping(event.data) for event in events if event.kind == "branch_selected"]
    assert [branch["branch"] for branch in branches] == ["continue"] * 5 + ["rephrase", "continue"]
    for branch in branches:
        exchange = exchanges[text(branch["checkpoint_id"])]
        assert branch["response_id"] == at(ends[text(exchange["call_id"])], "response", "id")
        assert branch["attempt_id"] == exchange["attempt_id"]
        for check in sequence(branch["checks"]):
            assert at(check, "context") == branch["context"]
            if at(check, "judgment"):
                assert at(check, "judgment", "call_id") in ends
    rollback = next(
        mapping(event.data) for event in events
        if event.kind == "conversation_synchronized"
        and at(event.data, "outcome", "conversation", "kind") == "restored"
    )
    assert rollback["cause_id"] == branches[5]["branch_id"]
    assert rollback["from_checkpoint_id"] == branches[5]["checkpoint_id"]
    assert rollback["checkpoint_id"] == branches[4]["checkpoint_id"]


def assert_evaluation_links(
    events: tuple[RunEvent, ...], starts: dict[str, dict[str, object]], result: Message,
) -> None:
    evaluations = [mapping(event.data) for event in events if event.kind == "evaluation_completed"]
    assert [evaluation["success"] for evaluation in evaluations] == [False, True]
    publications = [mapping(event.data) for event in events if event.kind == "result_published"]
    assert publications[0]["change"] == "accepted"
    assert publications[-1]["change"] == "replaced"
    assert at(publications[-1], "previous_evaluation", "result_id") == evaluations[0]["result_id"]
    for evaluation in evaluations:
        publication = next(
            item for item in publications if at(item, "result", "id") == evaluation["result_id"]
        )
        assert evaluation["checkpoint_id"] == publication["checkpoint_id"]
        assert evaluation["cause_id"] == publication["cause_id"]
        assert sum(start.get("result_id") == evaluation["result_id"] for start in starts.values()) == 1
    assert result.id == evaluations[-1]["result_id"]
