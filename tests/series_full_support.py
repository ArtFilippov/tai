"""Controlled mixed-outcome series and its cross-run isolation assertions."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from qwen_fakes import LocalLibraries
from series_support import TAXONOMY

from alert.records import AlertManifest
from alert.report import reconstruct_report
from app.configuration import RoleProfile
from app.series import SeriesResult
from core.chats.qwen_contracts import StoppingCriterion, Tensor
from tai import FITDExperiment, Message, ModelEvaluator, QwenChat

type ExperimentFactory = Callable[[RoleProfile, Message, Path | None], FITDExperiment]


@dataclass
class MixedSeries:
    lib: LocalLibraries
    manifest: AlertManifest
    users: list[FITDExperiment] = field(default_factory=list[FITDExperiment])
    profiles: list[RoleProfile] = field(default_factory=list[RoleProfile])
    target_calls: int = 0

    def create(self, factory: ExperimentFactory, profile: RoleProfile, base: Message, directory: Path) -> FITDExperiment:
        if self.users:
            previous = self.users[-1]
            if (run_id := previous.status().run_id) is not None:
                assert previous.wait(0), "next construction overlapped finalization"
                assert (directory / run_id / "metadata.json").is_file()
            saved = json.loads((directory.parent / "index.json").read_text())
            assert len(saved["entries"]) == len(self.users), "index was not incremental"
        self.target_calls = 0
        assert base.text == self.manifest.records[len(self.users)].base_request
        user = factory(profile, base, directory)
        assert isinstance(user._settings.evaluator, ModelEvaluator)
        assert user._settings.evaluator is profile.evaluator.evaluator
        assert user.base is base
        self.users.append(user)
        self.profiles.append(profile)
        return user

    def generate(self, criteria: Sequence[StoppingCriterion]) -> None:
        if self.lib.records.requests[-1][0]["content"] == "Target":
            self.target_calls += 1
            if self.target_calls == 1:
                assert len(self.lib.records.requests[-1]) == 2, "previous Target chat leaked"
            if len(self.users) == 4 and self.target_calls == 2:
                raise RuntimeError("controlled middle failure")

    def decode(self, tokens: Tensor, **kwargs: object) -> str:
        role = self.lib.records.requests[-1][0]["content"]
        if role in {"Attacker", "Evaluator"}:
            assert len(self.lib.records.requests[-1]) == 2
        if role == "Evaluator":
            payload = json.loads(self.lib.records.requests[-1][-1]["content"].split("INPUT_JSON:\n")[1])
            assert payload["base_request"] == self.manifest.records[len(self.users) - 1].base_request
            assert payload["result"] == "yes"
            if len(self.users) == 3:
                return '{"success": "false"}'
            return json.dumps({"success": len(self.users) != 1})
        return "yes"

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        factory, start = RoleProfile.create_experiment, FITDExperiment.start

        def create(profile: RoleProfile, base: Message, artifacts_dir: Path) -> FITDExperiment:
            return self.create(factory, profile, base, artifacts_dir)

        def begin(user: FITDExperiment) -> None:
            if len(self.users) == 2:
                raise RuntimeError("controlled start failure")
            start(user)

        monkeypatch.setattr(RoleProfile, "create_experiment", create)
        monkeypatch.setattr(FITDExperiment, "start", begin)
        self.lib.actions.generate = self.generate
        monkeypatch.setattr(self.lib, "decode", self.decode)


def assert_isolation(scenario: MixedSeries) -> None:
    assert len(scenario.users) == len({id(profile) for profile in scenario.profiles}) == 32
    assert len({id(user._settings.evaluator) for user in scenario.users}) == 32
    assert len({id(chat) for profile in scenario.profiles for chat in profile.chats().values()}) == 96
    assert len({user.base.id for user in scenario.users}) == 32
    assert len(scenario.lib.records.loads) == 2  # One tokenizer + one model for all roles and runs.
    profile = scenario.profiles[0]
    assert isinstance(profile.target, QwenChat)
    assert isinstance(profile.attacker, QwenChat)
    assert isinstance(profile.evaluator.chat, QwenChat)
    assert profile.target._executor is profile.attacker._executor
    assert profile.target._executor is profile.evaluator.chat._executor


def assert_mixed_outcomes(result: SeriesResult, scenario: MixedSeries) -> None:
    index = json.loads((result.directory / "index.json").read_text())
    assert [entry["id"] for entry in index["entries"]] == [record.id for record in scenario.manifest.records]
    assert [entry["state"] for entry in index["entries"][:5]] == ["COMPLETED", "IDLE", "FAILED", "FAILED", "COMPLETED"]
    missing = index["entries"][1]
    assert missing["run_id"] is None and missing["reason"] is None
    assert missing["error"]["phase"] == "start"
    assert all(value is None for value in missing["artifacts"].values())
    assert "ModelVerdictError" in index["entries"][2]["reason"]
    assert "middle failure" in index["entries"][3]["reason"]
    assert result.report["groups"]["hate"] == {"total": 8, "evaluated": 5, "successful": 4,
                                              "unevaluated": 3, "success_percent": 80.0}
    for entry, user in zip(index["entries"], scenario.users, strict=True):
        assert entry["started_at"] <= entry["finished_at"]
        assert entry["duration_seconds"] >= 0
        if entry["run_id"] is None:
            continue
        metadata = json.loads((result.directory / entry["artifacts"]["metadata.json"]).read_text())
        for key in ("state", "reason", "run_id", "result_id", "evaluation", "progress"):
            assert entry[key] == metadata[key]
        assert entry["base_id"] == metadata["settings"]["base"]["id"]
        assert metadata["settings"]["max_calls_per_conv"] == 20
        assert metadata["settings"]["generations"] == 1
        assert "base" not in metadata["settings"]["evaluator"]
        evaluations = [json.loads(event.dump())["data"] for event in user.journal.events() if event.kind == "evaluation_started"]
        assert all(event["base_id"] == entry["base_id"] for event in evaluations)
        assert metadata["final_settings"]["Evaluator"]["actual_dtype"] == "float16"
        assert user.wait(0)


def assert_series_files(result: SeriesResult, scenario: MixedSeries) -> None:
    before = len(scenario.lib.records.requests)
    output = result.directory / "rebuilt.json"
    assert reconstruct_report(result.directory / "index.json", output) == result.report
    assert output.read_bytes() == (result.directory / "report.json").read_bytes()
    assert len(scenario.lib.records.requests) == before
    assert json.loads((result.directory / "manifest.json").read_text()) == scenario.manifest.dump()
    assert (result.directory / "ALERT_taxonomy.md").read_bytes() == TAXONOMY.read_bytes()
    effective = json.loads((result.directory / "effective-profile.json").read_text())
    assert effective["fitd"]["max_calls_per_conv"] == 20
    assert effective["Evaluator"]["loaded"] is False
    assert json.loads((result.directory / "final-settings.json").read_text())["Evaluator"]["loaded"] is True
