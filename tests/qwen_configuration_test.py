"""Offline configuration, preflight and model-evaluator integration contracts."""

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import NoReturn, Protocol, cast

import pytest
from chats_assertions import at
from qwen_fakes import LocalFixture, local

from app.configuration import load_profile
from app.preflight import run_preflight
from core.chats import qwen
from tai import Message, ModelEvaluationError, ModelEvaluator, QwenChat, QwenError, UserState

__all__ = ["local"]


class InvalidQwenInput(Protocol):
    def __call__(self, **options: object) -> QwenChat: ...


def config_for(local: LocalFixture, tmp_path: Path, *, alert: bool = False) -> Path:
    _, kwargs, _ = local
    template = "examples/fitd-alert-qwen.toml" if alert else "examples/fitd-qwen.toml"
    text = Path(template).read_text(encoding="utf-8")
    text = text.replace("Qwen/Qwen2.5-0.5B-Instruct", kwargs["model"])
    text = text.replace("7ae557604adf67be50417f59c2c2f167def9a775", kwargs["revision"])
    text = text.replace("~/.cache/tai/huggingface", str(kwargs["cache_dir"]))
    config = tmp_path / "qwen.toml"
    config.write_text(text, encoding="utf-8")
    return config


def test_missing_local_files_gives_explicit_preparation_command(local: LocalFixture) -> None:
    lib, kwargs, snapshot = local
    (snapshot / "model.safetensors").unlink()
    chat = QwenChat(**kwargs)
    with pytest.raises(QwenError, match="hf download Qwen/test --revision"):
        chat.send(Message("request"))
    assert not lib.records.loads and not chat.history()


def test_missing_dependencies_and_cuda_do_not_fall_back(
    local: LocalFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib, kwargs, _ = local
    chat = QwenChat(**kwargs)

    def missing(_name: str) -> NoReturn:
        raise ImportError("missing optional library")

    monkeypatch.setattr(qwen, "import_module", missing)
    with pytest.raises(QwenError, match="uv sync --locked --extra qwen"):
        chat.send(Message("request"))
    monkeypatch.setattr(qwen, "import_module", lib.imports_for)
    lib.cuda_available = False
    with pytest.raises(QwenError, match="CUDA device cuda:0 is unavailable"):
        chat.send(Message("request"))
    assert not lib.records.loads and not chat.history()


def test_import_and_observation_do_not_import_optional_libraries() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "; ".join((
            "from app.configuration import load_profile", "from pathlib import Path",
            "import sys", "p = load_profile(Path('examples/fitd-alert-qwen.toml'))",
            "p.evaluator.settings()", "p.target.settings()",
            "assert not {'torch', 'transformers', 'huggingface_hub'} & sys.modules.keys()",
        ))], capture_output=True, text=True, timeout=10, check=False,
    )
    assert not completed.returncode, completed.stderr


def test_preflight_uses_public_chats_and_reports_effective_cuda_settings(
    local: LocalFixture, tmp_path: Path,
) -> None:
    lib, _, _ = local
    profile = load_profile(config_for(local, tmp_path))
    lib.responses.extend(["A real-shaped answer", "READY"])
    report = tmp_path / "preflight.json"
    assert not run_preflight(profile, report)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["ready"] is True
    assert [r["response"] for r in data["roles"]] == ["A real-shaped answer", "READY", None]
    assert data["roles"][2]["ok"] is True
    assert data["roles"][2]["success"] is False
    assert len({r["conversation_id"] for r in data["roles"][:2]}) == 2
    assert all(r["final_settings"]["actual_dtype"] == "float16" for r in data["roles"][:2])
    assert all(r["elapsed_seconds"] >= 0 for r in data["roles"])
    assert len(lib.records.generations) == 2 and len(lib.records.loads) == 2


@pytest.mark.parametrize("field", [
    'base_url="https://example.com"', 'api_key="forbidden"', 'authorization="forbidden"',
    'api_key_env="FORBIDDEN"', 'authorization_env="FORBIDDEN"',
    "opencode_session=true", "timeout=1", "max_tokens=512",
])
def test_qwen_rejects_http_fields_before_execution(
    local: LocalFixture, tmp_path: Path, field: str,
) -> None:
    lib, _, _ = local
    config = config_for(local, tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace("[target]\n", f"[target]\n{field}\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target"):
        load_profile(config)
    assert not lib.records.imports


@pytest.mark.parametrize("generation", [
    {"max_tokens": 10}, {"stream": False}, {"n": 1}, {"num_return_sequences": 2},
    {"past_key_values": []}, {"max_new_tokens": 0}, {"max_new_tokens": True},
    {"do_sample": 1}, {"temperature": 0.5}, {"do_sample": True, "temperature": 0},
    {"do_sample": True, "top_p": 1.1}, {"do_sample": True, "top_k": -1},
    {"repetition_penalty": math.nan}, {"do_sample": True, "top_k": True},
])
def test_invalid_generation_parameters_fail_before_import(
    local: LocalFixture, generation: dict[str, object],
) -> None:
    lib, kwargs, _ = local
    with pytest.raises(ValueError):
        QwenChat(**kwargs, generation=generation)
    assert not lib.records.imports


@pytest.mark.parametrize("override", [
    {"device": "cpu"}, {"device": "cuda"}, {"revision": "main"}, {"dtype": "auto"},
    {"cache_dir": ""}, {"system": 42}, {"model": "../local"},
])
def test_invalid_local_configuration_fails_before_import(
    local: LocalFixture, override: dict[str, object],
) -> None:
    lib, kwargs, _ = local
    # Deliberately cross the runtime validation boundary with malformed options.
    construct = cast(InvalidQwenInput, QwenChat)
    with pytest.raises(ValueError):
        construct(**{**kwargs, **override})
    assert not lib.records.imports


def test_backend_choice_and_http_compatibility(local: LocalFixture, tmp_path: Path) -> None:
    lib, _, _ = local
    config = config_for(local, tmp_path)
    text = config.read_text(encoding="utf-8")
    for backend in ("other", 12):
        config.write_text(
            text.replace('backend = "qwen"', f"backend = {json.dumps(backend)}"), encoding="utf-8",
        )
        with pytest.raises(ValueError, match="backend"):
            load_profile(config)
    for explicit in ("", 'backend="openai"'):
        config.write_text(f"""[target]
{explicit}
base_url="http://localhost:8080/v1"
model="http-model"
[attacker]
base_url="http://localhost:8080/v1"
model="http-model"
[evaluator]
type="contains_text"
criterion="key"
""", encoding="utf-8")
        assert at(load_profile(config).target.settings(), "backend") == "openai"
    assert not lib.records.imports


def test_three_role_preflight_and_real_fitd_reuse_one_load_across_base_requests(
    local: LocalFixture, tmp_path: Path,
) -> None:
    lib, _, snapshot = local
    profile = load_profile(config_for(local, tmp_path, alert=True))
    lib.responses.extend(["READY", "READY", '{"success": false}'])
    report = tmp_path / "preflight.json"
    assert not run_preflight(profile, report)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert [r["role"] for r in data["roles"]] == ["Target", "Attacker", "Evaluator"]
    assert all(r["final_settings"]["actual_dtype"] == "float16" for r in data["roles"][:2])
    assert data["roles"][2]["final_settings"]["chat"]["actual_dtype"] == "float16"
    assert len({r["conversation_id"] for r in data["roles"]}) == 3
    for base in (Message("First base"), Message("Second base")):
        lib.responses.extend(["Initial", "L1", "L2", "L3"])
        for response in ("One", "Two", "Three", "Four", "Presented"):
            lib.responses.extend([response, "yes"])
        lib.responses.append('{"success": true}')
        user = profile.create_experiment(base, tmp_path / "runs")
        user.start()
        assert user.wait(10)
        status = user.status()
        assert status.state is UserState.COMPLETED, status.reason
        assert status.evaluation is not None and status.evaluation.success is True
        assert status.progress.evaluator_calls == 1
        assert status.progress.max_calls_per_conv == 20
        assert len(lib.records.requests[-1]) == 2
        sent = json.loads(lib.records.requests[-1][-1]["content"].split("INPUT_JSON:\n")[1])
        assert sent == {"base_request": base.to_string(), "result": "Presented"}
    assert lib.records.loads == [("tokenizer", str(snapshot)), ("model", str(snapshot))]
    assert len(lib.records.generations) == 33
    assert not lib.responses


def test_evaluator_qwen_context_overflow_is_not_truncated_or_scored(local: LocalFixture) -> None:
    lib, kwargs, _ = local
    lib.input_length = 32760
    chat = QwenChat(**kwargs, generation={"max_new_tokens": 32})
    evaluator = ModelEvaluator(chat)
    full = "Complete result π\n" * 3000
    with pytest.raises(ModelEvaluationError, match="QwenContextError.*no messages were truncated"):
        evaluator.evaluate(Message("Explicit base"), Message(full))
    assert not lib.records.generations and not chat.history()
    sent = json.loads(lib.records.requests[-1][-1]["content"].split("INPUT_JSON:\n")[1])
    assert sent == {"base_request": "Explicit base", "result": full}
