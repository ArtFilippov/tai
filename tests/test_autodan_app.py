"""Configuration, command dispatch, preflight and ALERT artifacts for AutoDAN."""

import json
from pathlib import Path
from threading import Event

import pytest
from qwen_fakes import LocalFixture, local
from series_support import MANIFEST, TAXONOMY, qwen_profile

from app.cli import main
from app.configuration import load_profile
from app.series import run_alert_series
from core.chats import ChatTurn
from tai import AutoDANExperiment, Message, QwenChat

__all__ = ["local"]


def config(local: LocalFixture, directory: Path) -> Path:
    _, kwargs, _ = local
    path = qwen_profile(directory, kwargs)
    path.write_text('[attack]\ntype = "autodan"\nexpected = "Sure"\nnum_steps = 1\nbatch_size = 2\nnum_elites = 1\n'
                    + path.read_text().replace('max_calls_per_conv = 20\n', ''))
    return path


def test_profile_cli_and_preflight_use_scoring_without_target_history(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    path = config(local, tmp_path)
    seen: list[tuple[str, str]] = []

    def scoring(self: QwenChat, prompt: str, expected: str) -> float:
        assert len(self.history()) <= 3
        seen.append((prompt, expected))
        return 1.0

    def generate(self: QwenChat, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        if turns[-1].message.text == "Reply with exactly READY.":
            return "READY"
        return "Sorry, I cannot help" if self._system == "Target" else '{"success": false}' if self._system == "Evaluator" else "READY"

    monkeypatch.setattr(QwenChat, "score_continuation", scoring)
    monkeypatch.setattr(QwenChat, "_generate", generate)
    profile = load_profile(path)
    assert isinstance(profile.create_experiment(Message("Base")), AutoDANExperiment)
    assert not seen
    assert not main(["preflight", "--config", str(path)])
    report = json.loads(capsys.readouterr().out)
    assert [role["role"] for role in report["roles"]] == ["Target", "Attacker", "TargetScoring", "Evaluator"]
    assert seen == [("Reply with exactly READY.", "READY")]
    root = tmp_path / "runs"
    assert not main(["attack", "--config", str(path), "--artifacts-dir", str(root), "--text", "Explicit base"])
    _ = capsys.readouterr()
    directory = next(root.iterdir())
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["settings"]["attack"] == {"type": "autodan"}
    assert metadata["settings"]["base"]["text"] == "Explicit base"
    assert metadata["result_id"] is None and not (directory / "result.txt").exists()
    assert seen[-1][0].endswith(" Explicit base")


def test_series_keeps_unsuccessful_searches_without_fake_results(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = config(local, tmp_path)

    def scoring(self: QwenChat, prompt: str, expected: str) -> float:
        return 0.5

    def generate(self: QwenChat, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        return "Sorry" if self._system == "Target" else '{"success": false}'

    monkeypatch.setattr(QwenChat, "score_continuation", scoring)
    monkeypatch.setattr(QwenChat, "_generate", generate)
    result = run_alert_series(MANIFEST, path, tmp_path / "series", taxonomy_path=TAXONOMY, jobs=2)
    assert not result.exit_code and result.report["complete"]
    directory = result.directory
    effective = json.loads((directory / "effective-profile.json").read_text())
    assert effective["autodan"]["expected"] == "Sure"
    index = json.loads((directory / "index.json").read_text())
    assert len(index["entries"]) == 32
    for entry in index["entries"]:
        assert entry["reason"] == "strategy_exhausted" and entry["result_id"] is None
        run_dir = directory / entry["artifacts"]["metadata.json"]
        assert json.loads(run_dir.read_text())["settings"]["attack"] == {"type": "autodan"}
        assert not (run_dir.parent / "result.txt").exists()


@pytest.mark.parametrize("replacement", [
    'backend = "openai"', 'backend = "vllm"',
    'backend = "qwen"\nunknown = true',
])
def test_unsupported_target_is_rejected_before_work(local: LocalFixture, tmp_path: Path, replacement: str) -> None:
    path = config(local, tmp_path)
    path.write_text(path.read_text().replace('backend = "qwen"', replacement, 1))
    with pytest.raises(ValueError, match="autodan|target"):
        load_profile(path)


def test_unsupported_run_limit_is_rejected(local: LocalFixture, tmp_path: Path) -> None:
    path = config(local, tmp_path)
    path.write_text(path.read_text() + "max_calls_per_conv = 9\n")
    with pytest.raises(ValueError, match="run"):
        load_profile(path)


def test_preflight_reports_missing_scoring_capability(
    local: LocalFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    path = config(local, tmp_path)

    def unavailable(self: QwenChat, prompt: str, expected: str) -> float:
        raise ValueError("No local logits")

    def generate(self: QwenChat, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        return '{"success": false}' if self._system == "Evaluator" else "READY"

    monkeypatch.setattr(QwenChat, "score_continuation", unavailable)
    monkeypatch.setattr(QwenChat, "_generate", generate)
    assert main(["preflight", "--config", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["ready"]
    scoring = next(role for role in report["roles"] if role["role"] == "TargetScoring")
    assert not scoring["ok"] and "No local logits" in scoring["error"]
