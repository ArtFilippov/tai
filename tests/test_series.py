"""Real pinned FITD, shared Qwen executor, controlled offline library responses."""

import json
import signal
from pathlib import Path

import pytest
from qwen_fakes import LocalFixture
from series_full_support import MixedSeries, assert_isolation, assert_mixed_outcomes, assert_series_files
from series_support import MANIFEST, TAXONOMY, positive_responses, qwen_profile

from alert import load_alert_manifest
from alert.report import reconstruct_report
from app.cli import main
from app.series import run_alert_series
from tai import RunEvent, RunJournal

pytest_plugins = ("qwen_fakes",)


def test_series_captures_journal_failure_and_continues_after_release(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib, kwargs, _ = local
    config = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    write_event = RunJournal._write_event
    failed = False

    def write(journal: RunJournal, event: RunEvent) -> None:
        nonlocal failed
        data = event.data
        if not failed and event.kind == "call_started" and isinstance(data, dict) and data["role"] == "Target":
            failed = True
            raise OSError("journal unavailable")
        write_event(journal, event)

    monkeypatch.setattr(RunJournal, "_write_event", write)
    result = run_alert_series(MANIFEST, config, tmp_path / "runs", taxonomy_path=TAXONOMY, jobs=1)
    index = json.loads((result.directory / "index.json").read_text())
    assert failed and result.exit_code == 1
    first = index["entries"][0]
    assert first["state"] == "FAILED"
    assert first["error"] == {"phase": "journal", "type": "OSError", "message": "journal unavailable"}
    assert not first["progress"]["target_calls"]
    assert first["progress"]["attacker_calls"] == 4
    assert all(entry["state"] == "COMPLETED" for entry in index["entries"][1:])
    assert len(index["entries"]) == 32
    assert reconstruct_report(result.directory / "index.json") == result.report


def test_full_series_continues_in_order_isolates_bases_and_reuses_weights(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib, kwargs, _ = local
    config = qwen_profile(tmp_path, kwargs)
    scenario = MixedSeries(lib, load_alert_manifest(MANIFEST, taxonomy_path=TAXONOMY))
    scenario.install(monkeypatch)
    previous_handler = signal.getsignal(signal.SIGINT)
    result = run_alert_series(MANIFEST, config, tmp_path / "runs", taxonomy_path=TAXONOMY, jobs=1)
    assert signal.getsignal(signal.SIGINT) is previous_handler
    assert result.exit_code == 1 and result.report["complete"] is True
    assert_isolation(scenario)
    assert_mixed_outcomes(result, scenario)
    assert_series_files(result, scenario)


@pytest.mark.parametrize("rounds", [1, 200])
def test_repeat_series_has_unique_space_and_cli_reconstructs_without_inference(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    rounds: int,
) -> None:
    lib, kwargs, _ = local
    config = qwen_profile(tmp_path, kwargs)
    config.write_text(config.read_text().replace("max_calls_per_conv = 20", f"max_calls_per_conv = {rounds}"))
    positive_responses(lib, monkeypatch)
    args = ["alert-series", "--manifest", str(MANIFEST), "--profile", str(config),
            "--artifacts-dir", str(tmp_path / "runs"), "--taxonomy", str(TAXONOMY)]
    assert not main(args)
    first_output = json.loads(capsys.readouterr().out)
    first = tmp_path / "runs" / first_output["series_id"]
    effective = json.loads((first / "effective-profile.json").read_text())
    assert effective["fitd"]["max_calls_per_conv"] == rounds
    index = json.loads((first / "index.json").read_text())
    assert len(index["entries"]) == 32
    assert all(entry["progress"]["max_calls_per_conv"] == rounds for entry in index["entries"])
    old = {path: path.read_bytes() for path in first.rglob("*") if path.is_file()}
    assert not main(args)
    second_output = json.loads(capsys.readouterr().out)
    assert second_output["series_id"] != first_output["series_id"]
    assert all(path.read_bytes() == data for path, data in old.items())
    calls = len(lib.records.requests)
    assert not main(["alert-report", "--index", str(first / "index.json")])
    assert json.loads(capsys.readouterr().out) == first_output
    relocated = tmp_path / "relocated-series"
    first.rename(relocated)
    assert not main(["alert-report", "--index", str(relocated / "index.json")])
    assert json.loads(capsys.readouterr().out) == first_output
    _assert_relocated_links(relocated)
    assert len(lib.records.requests) == calls


def _assert_relocated_links(relocated: Path) -> None:
    index = json.loads((relocated / "index.json").read_text())
    for entry in index["entries"]:
        for name, link in entry["artifacts"].items():
            assert link is not None
            if name == "directory":
                assert (relocated / link).is_dir()
            else:
                assert (relocated / link).is_file()
