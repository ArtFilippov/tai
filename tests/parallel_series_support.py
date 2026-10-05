"""Gated real FITD participants and persisted snapshots for parallel acceptance."""

import json
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from threading import Event, Lock

import pytest

from alert.accounting import SeriesReport
from alert.values import array_value, object_value
from app.configuration import RoleProfile
from app.series_storage import SeriesFiles
from core.chats import ChatTurn, LLMChat
from core.evaluation.assembly import EvaluatorSetup


@dataclass
class ParallelScenario:
    size: InitVar[int] = 32
    entered: list[Event] = field(init=False)
    release: list[Event] = field(init=False)
    saved: list[Event] = field(init=False)
    profiles: list[RoleProfile] = field(default_factory=list[RoleProfile])
    snapshots: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    lock: Lock = field(default_factory=Lock)
    stopped: Event = field(default_factory=Event)

    def __post_init__(self, size: int) -> None:
        self.entered = [Event() for _ in range(size)]
        self.release = [Event() for _ in range(size)]
        self.saved = [Event() for _ in range(size)]

    def fresh(self, _profile: RoleProfile) -> RoleProfile:
        sequence = len(self.profiles)

        def target(_turns: tuple[ChatTurn, ...]) -> str:
            self.entered[sequence].set()
            assert self.release[sequence].wait(10)
            return f"Accepted answer {sequence}"

        replies = iter(["Initial", "L1", "L2", "L3", *(["yes"] * 20)])
        profile = RoleProfile(
            LLMChat(target, system="Target"), LLMChat(lambda turns: next(replies), system="Attacker"),
            EvaluatorSetup.model(LLMChat(lambda turns: '{"success":true}', system="Evaluator")), 20,
        )
        self.profiles.append(profile)
        return profile

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seed = RoleProfile(LLMChat(str), LLMChat(str), EvaluatorSetup.contains_text("unused"), 20)
        persist = SeriesFiles.persist

        def load(*_args: object, **_kwargs: object) -> RoleProfile:
            return seed

        def fresh(profile: RoleProfile) -> RoleProfile:
            return self.fresh(profile)

        def save(store: SeriesFiles, index: dict[str, object], report: SeriesReport) -> None:
            persist(store, index, report)
            if len(self.entered) == 32:
                self.record_snapshot(json.loads((store.directory / "index.json").read_text()))
            else:
                self.record_snapshot(index)

        monkeypatch.setattr("app.series.load_profile", load)
        monkeypatch.setattr(RoleProfile, "fresh", fresh)
        monkeypatch.setattr(SeriesFiles, "persist", save)

    def record_snapshot(self, snapshot: dict[str, object]) -> None:
        entries = array_value(snapshot["entries"], "entries")
        with self.lock:
            if len(self.entered) == 32:
                self.snapshots.append(snapshot)
            else:
                self.snapshots.append({"active": sum(object_value(value, "entry")["phase"] == "active"
                                                     for value in entries)})
        for i, value in enumerate(entries):
            if object_value(value, "entry")["phase"] == "finished":
                self.saved[i].set()
        if snapshot["interrupted"]:
            self.stopped.set()

    def reverse_completion(self, jobs: int) -> None:
        for event in self.entered[:jobs]:
            assert event.wait(5)
        assert len(self.profiles) == jobs
        self.release[jobs - 1].set()
        assert self.saved[jobs - 1].wait(5)
        if jobs < 32:
            assert self.entered[jobs].wait(5)
        if jobs > 1:
            assert not self.saved[0].is_set()

    def release_all(self) -> None:
        for event in self.release:
            event.set()

    def assert_independent(self) -> None:
        assert len({id(chat) for profile in self.profiles for chat in profile.chats().values()}) == 3 * len(self.profiles)
        assert len({id(profile.evaluator.evaluator) for profile in self.profiles}) == len(self.profiles)


def saved_index(directory: Path) -> dict[str, object]:
    return json.loads((directory / "index.json").read_text())
