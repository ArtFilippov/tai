"""Shared controlled Qwen profile and source paths for series acceptance tests."""

import json
from pathlib import Path

import pytest
from qwen_fakes import LocalLibraries, LocalOptions

from core.chats.qwen_contracts import Tensor

MANIFEST = Path("examples/alert-manifest.json")
TAXONOMY = Path("ALERT_taxonomy.md")


def qwen_profile(tmp_path: Path, kwargs: LocalOptions) -> Path:
    path = tmp_path / "profile.toml"
    path.write_text("\n".join(
        f'''[{table}]
backend = "qwen"
model = {json.dumps(kwargs["model"])}
revision = {json.dumps(kwargs["revision"])}
cache_dir = {json.dumps(str(kwargs["cache_dir"]))}
device = "cuda:0"
system = "{role}"
'''
        for table, role in (("target", "Target"), ("attacker", "Attacker"), ("evaluator.chat", "Evaluator"))
    ) + '''
[evaluator]
type = "model"
rubric_version = "alert-safety-v1"
[run]
generations = 1
max_calls_per_conv = 20
''')
    return path


def positive_responses(lib: LocalLibraries, monkeypatch: pytest.MonkeyPatch) -> None:
    def decode(tokens: Tensor, **kwargs: object) -> str:
        return '{"success": true}' if lib.records.requests[-1][0]["content"] == "Evaluator" else "yes"

    monkeypatch.setattr(lib, "decode", decode)
