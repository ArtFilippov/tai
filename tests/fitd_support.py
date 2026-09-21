"""Typed scripted participants and lifecycle assertions shared by FITD tests."""

import json
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from threading import Event

from tai import (
    ChatTurn, ContainsTextEvaluator, Evaluation, FITDExperiment, Message, User, capture_run, save_run,
)

type Turns = tuple[ChatTurn, ...]


class ScriptedModel:
    def __init__(self, responses: Iterable[str | Exception]) -> None:
        super().__init__()
        self.responses = iter(responses)
        self.requests: list[Turns] = []

    def __call__(self, turns: Turns) -> str:
        self.requests.append(turns)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingEvaluator:
    def __init__(self, criterion: str = "CONTROL") -> None:
        super().__init__()
        self.evaluator = ContainsTextEvaluator(criterion)
        self.results: list[Message] = []

    def evaluate(self, base: Message, result: Message) -> bool:
        self.results.append(result)
        return self.evaluator.evaluate(base, result)


def presented(user: User) -> Message:
    result = user.result()
    assert result is not None
    return result


def evaluation(user: User) -> Evaluation:
    value = user.status().evaluation
    assert value is not None
    return value


def run_id(user: User) -> str:
    value = user.status().run_id
    assert value is not None
    return value


def reason(user: User) -> str:
    value = user.status().reason
    assert value is not None
    return value


@contextmanager
def releasing(user: User, *signals: Event) -> Generator[None]:
    try:
        yield
    finally:
        for signal in signals:
            signal.set()
        assert user.wait(10)


def read_and_assert_final_artifacts(user: FITDExperiment, root: Path):
    save_run(capture_run(user, journal=user.journal), root)
    directory = root / run_id(user)
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    metadata = json.loads((directory / "metadata.json").read_text())
    assert events == [json.loads(event.dump()) for event in user.journal.events()]
    assert events[-1]["kind"] == "run_finished"
    assert metadata["state"] == user.status().state.value
    assert events[-1]["data"]["state"] == metadata["state"]
    assert events[-1]["data"]["reason"] == metadata["reason"] == user.status().reason
    if (result := user.result()) is None:
        assert metadata["result"] is None
        assert metadata["result_id"] is None
        assert not (directory / "result.txt").exists()
    else:
        assert metadata["result"] == json.loads(result.dump())
        assert metadata["result_id"] == result.id
        assert (directory / "result.txt").read_text() == result.to_string()
    return events, metadata
