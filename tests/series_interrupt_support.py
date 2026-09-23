"""Typed fault injection at series admission, finalization, and storage boundaries."""

import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

import pytest

from alert.values import array_value, object_value
from app import experiments, series_storage
from app.configuration import RoleProfile
from attacks.shared import execution
from attacks.shared.execution import Execution
from core.chats import LLMChat
from tai import FITDExperiment, Message, RunRecord, UserState


@dataclass
class InterruptRace:
    boundary: str
    raised: Event = field(default_factory=Event)

    def interrupt(self) -> None:
        if not self.raised.is_set():
            self.raised.set()
            # A real process signal, also when called from the FITD worker.
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGINT)

    def install_lifecycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        factory, start = RoleProfile.create_experiment, FITDExperiment.start

        def create(profile: RoleProfile, base: Message, directory: Path) -> FITDExperiment:
            user = factory(profile, base, directory)
            assert isinstance(user, FITDExperiment)
            if self.boundary == "construction":
                self.interrupt()
            return user

        def begin(user: FITDExperiment) -> None:
            if self.boundary == "before_start":
                self.interrupt()
            start(user)
            if self.boundary == "after_start":
                self.interrupt()

        monkeypatch.setattr(RoleProfile, "create_experiment", create)
        monkeypatch.setattr(FITDExperiment, "start", begin)
        self._install_finalization(monkeypatch)

    def _install_finalization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        finalize = Execution.finalize

        def finish(worker: Execution, state: UserState, reason: str) -> None:
            if self.boundary == "finalization":
                self.interrupt()
                assert worker._state.stop_requested.wait(5)
            finalize(worker, state, reason)

        monkeypatch.setattr(Execution, "finalize", finish)

    def _interrupt_save(self, path: Path, value: object) -> None:
        if path.name != "index.json":
            return
        index = object_value(value, "index")
        count = sum(bool(object_value(entry, "entry").get("finished_at")) for entry in array_value(index["entries"], "entries"))
        boundaries = {"initial_save": not count, "first_save": count == 1,
                      "last_save": count == 32 and bool(index["finished_at"])}
        if boundaries.get(self.boundary, False):
            self.interrupt()

    def install_storage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        write, save_run = series_storage.write_json, experiments.save_run

        def persist(path: Path, value: object) -> None:
            write(path, value)
            self._interrupt_save(path, value)

        def persist_run(record: RunRecord, directory: str | Path) -> Path:
            if self.boundary == "run_save":
                assert record.snapshot.status.state == "COMPLETED"
                self.interrupt()
            return save_run(record, directory)

        monkeypatch.setattr(series_storage, "write_json", persist)
        monkeypatch.setattr(experiments, "save_run", persist_run)


@dataclass
class AdmissionFaults:
    phase: str
    users: list[FITDExperiment] = field(default_factory=list[FITDExperiment])
    constructed: int = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        factory, start = RoleProfile.create_experiment, FITDExperiment.start

        def create(profile: RoleProfile, base: Message, directory: Path) -> FITDExperiment:
            self.constructed += 1
            if self.users and self.users[-1].status().run_id is not None:
                assert self.users[-1].wait(0)
            if self.phase == "construction" and self.constructed == 2:
                raise RuntimeError("factory unavailable")
            user = factory(profile, base, directory)
            assert isinstance(user, FITDExperiment)
            self.users.append(user)
            return user

        def begin(user: FITDExperiment) -> None:
            start(user)
            if self.phase == "after_start" and self.constructed == 2:
                raise RuntimeError("start admitted before raising")

        monkeypatch.setattr(RoleProfile, "create_experiment", create)
        monkeypatch.setattr(FITDExperiment, "start", begin)


@dataclass
class ReleaseGate:
    entered: Event = field(default_factory=Event)
    release: Event = field(default_factory=Event)
    checked: Event = field(default_factory=Event)
    users: list[FITDExperiment] = field(default_factory=list[FITDExperiment])

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        factory, release_chats = RoleProfile.create_experiment, execution.release_chats

        def create(profile: RoleProfile, base: Message, directory: Path) -> FITDExperiment:
            user = factory(profile, base, directory)
            assert isinstance(user, FITDExperiment)
            if self.users:
                assert self.checked.is_set() and self.users[-1].wait(0)
            self.users.append(user)
            return user

        def cleanup(chats: tuple[LLMChat, ...], owner: object) -> None:
            if not self.entered.is_set():
                self.entered.set()
                assert self.release.wait(5)
            release_chats(chats, owner)

        monkeypatch.setattr(RoleProfile, "create_experiment", create)
        monkeypatch.setattr(execution, "release_chats", cleanup)

    def _check(self) -> None:
        assert self.entered.wait(5)
        assert len(self.users) == 1
        assert self.users[0].status().state == "COMPLETED"
        assert not self.users[0].wait(0)
        self.checked.set()

    def control(self) -> None:
        try:
            self._check()
        finally:
            self.release.set()
