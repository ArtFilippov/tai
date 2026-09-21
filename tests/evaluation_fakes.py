"""Typed deterministic model responses for evaluator and artifact tests."""

from collections import deque
from collections.abc import Iterable, Sequence
from threading import Event

from core.chats import ChatTurn


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
        self.requests: list[tuple[ChatTurn, ...]] = []
        self.signals: list[Event] = []

    def __call__(self, turns: Sequence[ChatTurn], cancellation: Event | None = None) -> str:
        self.requests.append(tuple(turns))
        if cancellation is not None:
            self.signals.append(cancellation)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        if isinstance(response, HeldResponse):
            assert cancellation is not None
            return response(cancellation)
        return response


def script_success(
    target: ScriptedModel, attacker: ScriptedModel, marker: str, initial: str = "Initial"
) -> None:
    target.responses.extend(["One", "Two", "Three", "Four", f"CONTROL {marker}"])
    attacker.responses.extend([initial, "L1", "L2", "L3"] + ["yes"] * 5)
