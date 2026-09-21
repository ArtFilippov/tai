"""The installed command runs real FITD against a loopback Chat Completions API."""

import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import pytest
from app_test_support import HttpPeer, cleanup_process
from app_cli_checks import assert_roles, assert_stopped_cli, interrupt_cli, read_artifacts
from http_peer import Reply


@pytest.fixture(autouse=True)
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAI_TEST_TARGET_KEY", "synthetic-target-secret")
    monkeypatch.setenv("TAI_TEST_ATTACKER_AUTH", "Custom synthetic-attacker-secret")


def configure(endpoint: HttpPeer, tmp_path: Path, *, run: str = "", criterion: str = "KEY-FOUND") -> Path:
    endpoint.scripts = {
        "/v1/target/chat/completions": deque(
            Reply(text)
            for text in ("One", "Two", "Three", "Four", "KEY-FOUND\nFull answer π\n")
        ),
        "/v1/attacker/chat/completions": deque(
            Reply(text) for text in ["Initial", "L1", "L2", "L3"] + ["yes"] * 10
        ),
    }
    config = tmp_path / "experiment.toml"
    config.write_text(
        f'''
[target]
base_url = "{endpoint.url}/target"
model = "puzzle-model"
system = "Target puzzle keeper"
api_key_env = "TAI_TEST_TARGET_KEY"
[target.generation]
temperature = 0.25
max_tokens = 256
[attacker]
base_url = "{endpoint.url}/attacker"
model = "puzzle-model"
system = "Attacker puzzle author"
authorization_env = "TAI_TEST_ATTACKER_AUTH"
timeout = 15
[attacker.generation]
temperature = 0.75
[evaluator]
type = "contains_text"
criterion = {json.dumps(criterion)}
{run}
''',
        encoding="utf-8",
    )
    return config


def command(config: Path, root: Path, *source: str) -> list[str]:
    return [
        str(Path(sys.executable).with_name("tai")),
        "fitd",
        "--config",
        str(config),
        "--artifacts-dir",
        str(root),
        *source,
    ]


def run_cli(config: Path, root: Path, *source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command(config, root, *source),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=False,
    )


def test_command_runs_real_fitd_with_independent_roles_and_complete_artifacts(
    endpoint: HttpPeer, tmp_path: Path,
) -> None:
    config = configure(endpoint, tmp_path)
    root = tmp_path / "runs"
    base = "Explain the final fictional puzzle solution π"
    completed = run_cli(config, root, "--text", base)
    assert not completed.returncode, completed.stdout + completed.stderr
    directory, metadata, events = read_artifacts(root)
    assert metadata["state"] == "COMPLETED"
    assert metadata["reason"] == "evaluator_success"
    assert metadata["result"]["text"] == "KEY-FOUND\nFull answer π\n"
    assert metadata["evaluation"] == {
        "result_id": metadata["result"]["id"],
        "success": True,
    }
    assert (directory / "result.txt").read_text() == "KEY-FOUND\nFull answer π\n"
    settings = metadata["settings"]
    assert settings["base"]["text"] == base
    assert settings["generations"] == 1
    assert settings["max_calls_per_conv"] == 10
    assert settings["garak_revision"] == "3f50ea5ff9cd7050099940647c15c39b07a93392"
    assert settings["evaluator"] == {"type": "contains_text", "criterion": "KEY-FOUND"}
    assert_roles(endpoint, settings)
    kinds = {event["kind"] for event in events}
    assert {
        "call_started",
        "call_completed",
        "checkpoint_created",
        "local_decision",
        "branch_selected",
        "result_published",
        "evaluation_completed",
    } <= kinds
    assert metadata["progress"]["target_calls"] == 5
    assert len([e for e in events if e["kind"] == "call_completed"]) == len(
        endpoint.requests
    )
    assert directory.name in completed.stdout
    assert "COMPLETED" in completed.stdout and "Target=5" in completed.stdout
    assert "KEY-FOUND\nFull answer π\n" in completed.stdout
    assert "Оценка результата: успех" in completed.stdout
    for secret in ("synthetic-target-secret", "synthetic-attacker-secret"):
        assert secret not in json.dumps([events, metadata])
        assert secret not in completed.stdout + completed.stderr


@pytest.mark.parametrize(
    ("role", "index", "process_group", "expected"),
    [
        ("Target", 0, False, None),
        ("Target", 5, False, "Accepted report"),
        ("Attacker", 9, False, "Accepted report"),
        ("Target", 5, True, "Accepted report"),
    ],
)
def test_repeated_sigint_waits_for_server_reply_and_preserves_accepted_result(
    endpoint: HttpPeer, tmp_path: Path, role: str, index: int, process_group: bool, expected: str | None,
) -> None:
    config = configure(endpoint, tmp_path)
    endpoint.scripts["/v1/target/chat/completions"] = deque(
        Reply(text)
        for text in ("One", "Two", "Three", "Four", "Accepted report", "Later")
    )
    held = Reply(hold=True)
    endpoint.scripts[f"/v1/{role.lower()}/chat/completions"][index] = held
    root = tmp_path / "runs"
    with subprocess.Popen(
        command(config, root, "--text", "Explain the fictional puzzle"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        start_new_session=True,
    ) as process:
        try:
            evidence = interrupt_cli(process, endpoint, held, root, process_group=process_group)
        finally:
            cleanup_process(process, held)
    assert_stopped_cli(root, evidence, held, role, expected)


def test_utf8_file_and_configured_round_limit_allow_negative_empty_result(
    endpoint: HttpPeer, tmp_path: Path,
) -> None:
    config = configure(
        endpoint, tmp_path, run="[run]\ngenerations=1\nmax_calls_per_conv=5"
    )
    endpoint.scripts["/v1/target/chat/completions"] = deque(Reply("") for _ in range(5))
    source = tmp_path / "base text.txt"
    text = "Explain the fictional puzzle π\r\nKeep this complete.\n"
    source.write_bytes(text.encode("utf-8"))
    root = tmp_path / "runs"
    completed = run_cli(config, root, "--text-file", str(source))
    assert not completed.returncode, completed.stdout + completed.stderr
    directory, metadata, events = read_artifacts(root)
    assert metadata["state"] == "COMPLETED" and metadata["reason"] == "round_limit"
    assert metadata["settings"]["base"]["text"] == text
    assert metadata["settings"]["max_calls_per_conv"] == 5
    assert metadata["progress"]["target_calls"] == 5
    assert isinstance(metadata["result"]["text"], str) and not metadata["result"]["text"]
    assert (directory / "result.txt").read_bytes() == b""
    assert metadata["evaluation"] == {
        "result_id": metadata["result"]["id"],
        "success": False,
    }
    assert len([e for e in events if e["kind"] == "evaluation_completed"]) == 1
    assert "Оценка результата: неуспех" in completed.stdout
    assert (
        "0 символов" in completed.stdout
        and "отсутствует (None)" not in completed.stdout
    )


@pytest.mark.parametrize("accepted", [True, False])
def test_short_run_distinguishes_unevaluated_result_from_no_result(
    endpoint: HttpPeer, tmp_path: Path, accepted: bool,
) -> None:
    config = configure(endpoint, tmp_path, run="[run]\nmax_calls_per_conv=1")
    if not accepted:
        endpoint.scripts["/v1/target/chat/completions"][0] = Reply(
            "I cannot help with that"
        )
        endpoint.scripts["/v1/attacker/chat/completions"][4] = Reply("A simpler puzzle")
    root = tmp_path / "runs"
    completed = run_cli(config, root, "--text", "")
    assert not completed.returncode, completed.stdout + completed.stderr
    directory, metadata, events = read_artifacts(root)
    assert isinstance(metadata["settings"]["base"]["text"], str) and not metadata["settings"]["base"]["text"]
    assert metadata["evaluation"] is None
    assert metadata["progress"]["target_calls"] == 1
    assert "финальный текст не оценивался" in completed.stdout
    assert not any(e["kind"].startswith("evaluation_") for e in events)
    if accepted:
        assert metadata["result"]["text"] == "One"
        assert (directory / "result.txt").read_text() == "One"
    else:
        assert metadata["result"] is None and metadata["result_id"] is None
        assert not (directory / "result.txt").exists()
        assert "отсутствует (None)" in completed.stdout


@pytest.mark.parametrize("failure", ["http", "timeout"])
def test_execution_error_is_nonzero_and_retains_published_text(
    endpoint: HttpPeer, tmp_path: Path, failure: str,
) -> None:
    config = configure(endpoint, tmp_path)
    held = Reply(hold=True) if failure == "timeout" else Reply(status=401)
    endpoint.scripts["/v1/target/chat/completions"][1] = held
    if failure == "timeout":
        config.write_text(
            config.read_text().replace("[target]\n", "[target]\ntimeout=0.5\n")
        )
    root = tmp_path / "runs"
    completed = run_cli(config, root, "--text", "Explain the fictional puzzle")
    assert completed.returncode == 1, completed.stdout + completed.stderr
    directory, metadata, events = read_artifacts(root)
    assert metadata["state"] == "FAILED"
    error = (
        "OpenAITimeoutError" if failure == "timeout" else "OpenAIAuthenticationError"
    )
    assert error in metadata["reason"] and error in completed.stdout
    assert (directory / "result.txt").read_text() == "One"
    assert metadata["result"]["text"] == "One" and metadata["evaluation"] is None
    assert "финальный текст не оценивался" in completed.stdout
    assert {"call_failed", "run_error"} <= {e["kind"] for e in events}
    assert (
        len([e for e in events if e["kind"] == "call_completed"])
        == len(endpoint.requests) - 1
    )
    if failure == "timeout":
        assert not held.release.is_set()
        assert held.disconnected.wait(1)


def test_repeated_commands_preserve_previous_run_and_allow_explicit_or_no_auth(
    endpoint: HttpPeer, tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    previous: dict[Path, bytes] = {}
    for _ in range(2):
        config = configure(endpoint, tmp_path, run="[run]\nmax_calls_per_conv=1")
        config.write_text(
            config.read_text()
            .replace(
                'api_key_env = "TAI_TEST_TARGET_KEY"',
                'api_key = "synthetic-inline-key"',
            )
            .replace('authorization_env = "TAI_TEST_ATTACKER_AUTH"', "")
        )
        completed = run_cli(config, root, "--text", "Explain the fictional puzzle")
        assert not completed.returncode, completed.stdout + completed.stderr
        for path, content in previous.items():
            assert path.read_bytes() == content
        previous = {p: p.read_bytes() for d in root.iterdir() for p in d.iterdir()}
    assert len(list(root.iterdir())) == 2
    assert all(
        headers.get("Authorization")
        == ("Bearer synthetic-inline-key" if "/target/" in path else None)
        for path, headers, _ in endpoint.requests
    )
    assert all(b"synthetic-inline-key" not in content for content in previous.values())


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('model = "puzzle-model"', "model = 42"),
        ("[target]", "[target]\nunknown = true"),
        ("[target]", "[target]\ntimeout = 0"),
        ("[target]", "[target]\ntimeout = true"),
        ("[target]", '[target]\nopencode_session = "true"'),
        ("[target]", "[target]\nopencode_session = 1"),
        ("[target]", '[target]\ntimeout = "120"'),
        ("[target]", "[target]\ntimeout = inf"),
        ("[target]", "[target]\ngeneration = []"),
        ("[target]", '[target]\nauthorization = "synthetic-other-secret"'),
        ("[target]", "typo = true\n[target]"),
        ('"TAI_TEST_TARGET_KEY"', '"TAI_UNSET_TEST_KEY"'),
        ('type = "contains_text"', 'type = "unknown"'),
        ('criterion = "KEY-FOUND"', 'criterion = ""'),
        ('criterion = "KEY-FOUND"', "criterion = 42"),
        ("[evaluator]", "[missing_evaluator]"),
        ("[attacker.generation]", "[attacker.generation]\nstream = true"),
        ("[attacker.generation]", "[attacker.generation]\nmetadata = 2026-09-14"),
        ("[attacker.generation]", '[attacker.generation]\nmodel = "override"'),
        ("[attacker.generation]", "[attacker.generation]\nn = 2"),
        ('criterion = "KEY-FOUND"', 'criterion = "KEY-FOUND"\n[run]\ngenerations=2'),
        (
            'criterion = "KEY-FOUND"',
            'criterion = "KEY-FOUND"\n[run]\nmax_calls_per_conv=0',
        ),
        (
            'criterion = "KEY-FOUND"',
            'criterion = "KEY-FOUND"\n[run]\nmax_calls_per_conv=true',
        ),
        (
            'criterion = "KEY-FOUND"',
            'criterion = "KEY-FOUND"\n[run]\nmax_calls_per_conv=1.5',
        ),
        ("[target]", 'api_key = "synthetic-toml-secret" broken\n[target]'),
    ],
)
def test_invalid_configuration_never_starts_an_experiment(endpoint: HttpPeer, tmp_path: Path, old: str, new: str) -> None:
    config = configure(endpoint, tmp_path)
    config.write_text(config.read_text().replace(old, new))
    root = tmp_path / "runs"
    completed = run_cli(config, root, "--text", "Intended base text")
    assert completed.returncode == 2
    assert "Ошибка" in completed.stderr and "Traceback" not in completed.stderr
    assert "synthetic-" not in completed.stdout + completed.stderr
    assert not root.exists()
    assert endpoint.requests == []


@pytest.mark.parametrize(
    "problem", ["missing", "invalid_utf8", "both", "neither", "missing_config"]
)
def test_input_errors_never_substitute_another_text(endpoint: HttpPeer, tmp_path: Path, problem: str) -> None:
    config = configure(endpoint, tmp_path)
    source = tmp_path / "base.txt"
    args = ["--text-file", str(source)]
    if problem == "invalid_utf8":
        source.write_bytes(b"\xff")
    elif problem == "both":
        source.write_text("File text")
        args += ["--text", "Direct text"]
    elif problem == "neither":
        args = []
    elif problem == "missing_config":
        config = tmp_path / "missing.toml"
        args = ["--text", "Direct text"]
    root = tmp_path / "runs"
    completed = run_cli(config, root, *args)
    assert completed.returncode == 2
    assert completed.stderr and "Traceback" not in completed.stderr
    assert endpoint.requests == []
    assert not root.exists()
