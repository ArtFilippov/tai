"""Typed recordings of actual operation returns, never private strategy state."""

from collections import Counter, defaultdict
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from threading import Event
from typing import TypedDict, cast

import pytest
from fitd_support import ScriptedModel

from attacks.fitd.contracts import Continuation
from attacks.fitd.execution import Execution
from attacks.fitd.garak_adapter import FITDProbe
from attacks.fitd.native import Attempt, FITD
from attacks.fitd.target import FITDTarget
from core.runs.operations import (
    Checkpoint, ConversationChange, ConversationPoint, EvaluationResult, Exchange,
    LocalDecision, ResourcesReleased, ResultChange, RunStarted, Synchronization, TargetExchange, WorkerOutcome,
)
from tai import ContainsTextEvaluator, FITDExperiment, LLMChat, Message


class OperationValues(TypedDict):
    check_refusal: list[LocalDecision]
    check_alignment: list[LocalDecision]
    select_continuation: list[Continuation[Attempt]]
    save: list[ConversationPoint]
    exchange: list[TargetExchange]
    synchronize: list[Synchronization]
    accept: list[ResultChange]
    start: list[RunStarted]
    _send: list[Exchange]
    checkpoint: list[Checkpoint]
    restore: list[ConversationChange]
    reset: list[ConversationChange]
    publish: list[ResultChange]
    _evaluate: list[EvaluationResult]
    record_evaluation: list[EvaluationResult]
    _run: list[WorkerOutcome]
    _release_resources: list[ResourcesReleased]


class Observations:
    def __init__(self) -> None:
        super().__init__()
        self._values: defaultdict[str, list[object]] = defaultdict(list)
        self.errors: defaultdict[str, list[Exception]] = defaultdict(list)
        self.native: Counter[str] = Counter()
        self.finished = Event()

    @property
    def values(self) -> OperationValues:
        # install_observers binds precisely these named operations and records
        # their unmodified returns. The cast describes that reflection boundary.
        return cast(OperationValues, self._values)

    def wrap[**P, R](self, operation: Callable[P, R]) -> Callable[P, R]:
        @wraps(operation)
        def observe(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                value = operation(*args, **kwargs)
            except Exception as exc:
                self.errors[operation.__name__].append(exc)
                raise
            self._values[operation.__name__].append(value)
            if operation.__name__ == "_run":
                self.finished.set()
            return value
        return observe


def install_observers(monkeypatch: pytest.MonkeyPatch) -> Observations:
    observations = Observations()
    for owner, methods in (
        (FITDProbe, ("check_refusal", "check_alignment", "select_continuation")),
        (FITDTarget, ("save", "exchange", "synchronize", "accept")),
        (Execution, (
            "start", "_send", "checkpoint", "restore", "reset", "publish", "_evaluate",
            "record_evaluation", "_run", "_release_resources",
        )),
    ):
        for method in methods:
            original = cast(Callable[..., object], getattr(owner, method))
            monkeypatch.setattr(owner, method, observations.wrap(original))
    _count_native(monkeypatch, observations)
    return observations


def _count_native(monkeypatch: pytest.MonkeyPatch, observations: Observations) -> None:
    def count[**P, R](operation: Callable[P, R]) -> Callable[P, R]:
        @wraps(operation)
        def call(*args: P.args, **kwargs: P.kwargs) -> R:
            observations.native[operation.__name__] += 1
            return operation(*args, **kwargs)
        return call

    for method in ("_is_refusal", "_is_response_aligned", "_is_realign_prompt", "generate_next_turn_attempt"):
        original = cast(Callable[..., object], getattr(FITD, method))
        monkeypatch.setattr(FITD, method, count(original))


def puzzle(tmp_path: Path, targets: list[str | Exception], attackers: list[str], rounds: int):
    target = LLMChat(ScriptedModel(targets), system="Puzzle rules")
    attacker = LLMChat(ScriptedModel(attackers), system="Build puzzle requests")
    user = FITDExperiment(
        target, attacker, Message("Solve the fictional puzzle"), ContainsTextEvaluator("CONTROL"),
        journal_dir=tmp_path, max_calls_per_conv=rounds,
    )
    return user, target, attacker
