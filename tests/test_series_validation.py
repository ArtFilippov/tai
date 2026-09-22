"""All series inputs are validated before admitting any User work."""

import json
from pathlib import Path

import pytest
from qwen_fakes import LocalFixture
from series_support import MANIFEST, TAXONOMY, qwen_profile

from app.cli import main
from app.series import run_alert_series
from tai import FITDExperiment

pytest_plugins = ("qwen_fakes",)


def _corrupt_manifest(manifest: Path) -> None:
    data = json.loads(manifest.read_text())
    data["records"][-1]["base_request"] = "Changed"
    manifest.write_text(json.dumps(data))


def _corrupt_profile(profile: Path, problem: str) -> None:
    if problem == "evaluator":
        profile.write_bytes(Path("examples/fitd-qwen.toml").read_bytes())
    else:
        old, new = {
            "profile": ('system = "Evaluator"', "unknown = true"),
            "rounds": ("max_calls_per_conv = 20", "max_calls_per_conv = 0"),
        }[problem]
        profile.write_text(profile.read_text().replace(old, new))


def _invalid_paths(problem: str, profile: Path, manifest: Path, artifacts: Path) -> Path:
    if problem == "last_record":
        _corrupt_manifest(manifest)
    elif problem in {"profile", "rounds", "evaluator"}:
        _corrupt_profile(profile, problem)
    elif problem == "taxonomy":
        return profile.parent / "absent.md"
    else:
        artifacts.write_text("not a directory")
    return TAXONOMY


@pytest.mark.parametrize("problem", ["last_record", "profile", "rounds", "evaluator", "taxonomy", "artifacts"])
def test_all_inputs_are_validated_before_any_start(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str,
) -> None:
    lib, kwargs, _ = local
    profile = qwen_profile(tmp_path, kwargs)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(MANIFEST.read_bytes())
    artifacts = tmp_path / "runs"
    taxonomy = _invalid_paths(problem, profile, manifest, artifacts)

    def forbidden(user: FITDExperiment) -> None:
        pytest.fail("start on invalid input")

    monkeypatch.setattr(FITDExperiment, "start", forbidden)
    assert main(["alert-series", "--manifest", str(manifest), "--profile", str(profile),
                 "--artifacts-dir", str(artifacts), "--taxonomy", str(taxonomy)]) == 2
    assert lib.records.requests == [] and lib.records.loads == []
    if problem != "artifacts":
        assert not artifacts.exists()


@pytest.mark.parametrize("lines", [0, -1, True])
def test_invalid_line_budget_precedes_input_loading(tmp_path: Path, lines: int) -> None:
    with pytest.raises(ValueError, match="lines_per_attack"):
        run_alert_series(MANIFEST, tmp_path / "absent.toml", tmp_path / "runs", taxonomy_path=TAXONOMY, lines_per_attack=lines)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("lines", ["0", "-1", "text"])
def test_cli_rejects_invalid_line_budget(tmp_path: Path, lines: str) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["alert-series", "--manifest", str(MANIFEST), "--profile", str(tmp_path / "absent.toml"),
              "--artifacts-dir", str(tmp_path / "runs"), "--lines-per-attack", lines])
    assert caught.value.code == 2 and not (tmp_path / "runs").exists()
