"""Real pinned FITD, shared Qwen executor, controlled offline library responses."""

import errno
import json
import signal
from pathlib import Path

import pytest
from test_qwen import local as local

from tai import FITDUser, ModelEvaluator, RunJournal
from tai._config import RoleProfile
from tai.alert import load_alert_manifest
from tai.cli import main
from tai.series import run_alert_series
from tai.series_report import reconstruct_report

MANIFEST = Path("examples/alert-manifest.json")
TAXONOMY = Path("ALERT_taxonomy.md")


def qwen_profile(tmp_path, kwargs):
    path = tmp_path / "profile.toml"
    path.write_text(
        "\n".join(
            f'''[{table}]
backend = "qwen"
model = {json.dumps(kwargs["model"])}
revision = {json.dumps(kwargs["revision"])}
cache_dir = {json.dumps(str(kwargs["cache_dir"]))}
device = "cuda:0"
system = "{role}"
'''
            for table, role in [
                ("target", "Target"),
                ("attacker", "Attacker"),
                ("evaluator.chat", "Evaluator"),
            ]
        )
        + """
[evaluator]
type = "model"
rubric_version = "alert-safety-v1"
[run]
generations = 1
max_calls_per_conv = 20
"""
    )
    return path


def positive_responses(lib, monkeypatch):
    def decode(tokens, **kwargs):
        return (
            '{"success": true}'
            if lib.requests[-1][0]["content"] == "Evaluator"
            else "yes"
        )

    monkeypatch.setattr(lib, "decode", decode)


def test_series_captures_journal_failure_and_continues_after_release(
    local, tmp_path, monkeypatch
):
    lib, kwargs, _ = local
    config = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    write_event = RunJournal._write_event
    failed = False

    def write(journal, event):
        nonlocal failed
        if (
            not failed
            and event.kind == "call_started"
            and event.data["role"] == "Target"
        ):
            failed = True
            raise OSError("journal unavailable")
        return write_event(journal, event)

    monkeypatch.setattr(RunJournal, "_write_event", write)
    result = run_alert_series(
        MANIFEST, config, tmp_path / "runs", taxonomy_path=TAXONOMY
    )
    index = json.loads((result.directory / "index.json").read_text())
    assert failed and result.exit_code == 1
    first = index["entries"][0]
    assert first["state"] == "FAILED"
    assert first["error"] == {
        "phase": "journal",
        "type": "OSError",
        "message": "journal unavailable",
    }
    assert first["progress"]["target_calls"] == 0
    assert first["progress"]["attacker_calls"] == 4
    assert all(e["state"] == "COMPLETED" for e in index["entries"][1:])
    assert len(index["entries"]) == 32
    assert reconstruct_report(result.directory / "index.json") == result.report


def test_full_series_continues_in_order_isolates_bases_and_reuses_weights(
    local, tmp_path, monkeypatch
):
    lib, kwargs, _ = local
    config = qwen_profile(tmp_path, kwargs)
    manifest = load_alert_manifest(MANIFEST, taxonomy_path=TAXONOMY)
    users, profiles = [], []
    factory = RoleProfile.create_experiment
    start = FITDUser.start
    target_calls = 0

    def create(profile, base, artifacts_dir):
        nonlocal target_calls
        if users:
            previous = users[-1]
            if previous.status().run_id is not None:
                assert previous.wait(0), "next construction overlapped finalization"
                assert (
                    artifacts_dir / previous.status().run_id / "metadata.json"
                ).is_file()
            saved = json.loads((artifacts_dir.parent / "index.json").read_text())
            assert len(saved["entries"]) == len(users), "index was not incremental"
        target_calls = 0
        assert base.text == manifest.records[len(users)].base_request
        user = factory(profile, base, artifacts_dir)
        assert isinstance(user._evaluator, ModelEvaluator)
        assert user._evaluator.base is base
        users.append(user)
        profiles.append(profile)
        return user

    def begin(user):
        if len(users) == 2:
            raise RuntimeError("controlled start failure")
        start(user)

    def generate(criteria):
        nonlocal target_calls
        if lib.requests[-1][0]["content"] == "Target":
            target_calls += 1
            if target_calls == 1:
                assert len(lib.requests[-1]) == 2, "previous Target chat leaked"
            if len(users) == 4 and target_calls == 2:
                raise RuntimeError("controlled middle failure")

    def decode(tokens, **kwargs):
        role = lib.requests[-1][0]["content"]
        if role in ("Attacker", "Evaluator"):
            assert len(lib.requests[-1]) == 2
        if role == "Evaluator":
            payload = json.loads(
                lib.requests[-1][-1]["content"].split("INPUT_JSON:\n")[1]
            )
            assert (
                payload["base_request"] == manifest.records[len(users) - 1].base_request
            )
            assert payload["result"] == "yes"
            if len(users) == 3:
                return '{"success": "false"}'
            return json.dumps({"success": len(users) != 1})
        return "yes"

    monkeypatch.setattr(RoleProfile, "create_experiment", create)
    monkeypatch.setattr(FITDUser, "start", begin)
    lib.generate_action = generate
    monkeypatch.setattr(lib, "decode", decode)
    previous_handler = signal.getsignal(signal.SIGINT)
    result = run_alert_series(
        MANIFEST, config, tmp_path / "runs", taxonomy_path=TAXONOMY
    )
    assert signal.getsignal(signal.SIGINT) is previous_handler
    assert result.exit_code == 1 and result.report["complete"] is True
    assert len(users) == 32 and all(p is profiles[0] for p in profiles)
    assert len({id(u._evaluator) for u in users}) == 32
    assert len({u._base.id for u in users}) == 32
    assert len(lib.loads) == 2  # One tokenizer + one model for all roles and runs.
    profile = profiles[0]
    assert profile.target._executor is profile.attacker._executor
    assert profile.target._executor is profile.evaluator.chat._executor
    index = json.loads((result.directory / "index.json").read_text())
    assert [entry["id"] for entry in index["entries"]] == [
        r.id for r in manifest.records
    ]
    assert [e["state"] for e in index["entries"][:5]] == [
        "COMPLETED",
        "IDLE",
        "FAILED",
        "FAILED",
        "COMPLETED",
    ]
    missing = index["entries"][1]
    assert missing["run_id"] is None and missing["reason"] is None
    assert missing["error"]["phase"] == "start"
    assert all(value is None for value in missing["artifacts"].values())
    assert "ModelVerdictError" in index["entries"][2]["reason"]
    assert "middle failure" in index["entries"][3]["reason"]
    assert result.report["groups"]["hate"] == {
        "total": 8,
        "evaluated": 5,
        "successful": 4,
        "unevaluated": 3,
        "success_percent": 80.0,
    }
    for entry, user in zip(index["entries"], users, strict=True):
        assert entry["started_at"] <= entry["finished_at"]
        assert entry["duration_seconds"] >= 0
        if entry["run_id"] is None:
            continue
        metadata = json.loads(
            (result.directory / entry["artifacts"]["metadata.json"]).read_text()
        )
        for key in ("state", "reason", "run_id", "result_id", "evaluation", "progress"):
            assert entry[key] == metadata[key]
        assert entry["base_id"] == metadata["settings"]["base"]["id"]
        assert metadata["settings"]["max_calls_per_conv"] == 20
        assert metadata["settings"]["generations"] == 1
        assert metadata["settings"]["evaluator"]["base"]["id"] == entry["base_id"]
        assert metadata["final_settings"]["Evaluator"]["actual_dtype"] == "float16"
        assert user.wait(0)
    before = len(lib.requests)
    output = result.directory / "rebuilt.json"
    assert reconstruct_report(result.directory / "index.json", output) == result.report
    assert output.read_bytes() == (result.directory / "report.json").read_bytes()
    assert len(lib.requests) == before
    assert (
        json.loads((result.directory / "manifest.json").read_text()) == manifest.dump()
    )
    assert (
        result.directory / "ALERT_taxonomy.md"
    ).read_bytes() == TAXONOMY.read_bytes()
    effective = json.loads((result.directory / "effective-profile.json").read_text())
    assert effective["fitd"]["max_calls_per_conv"] == 20
    assert effective["Evaluator"]["loaded"] is False
    assert (
        json.loads((result.directory / "final-settings.json").read_text())["Evaluator"][
            "loaded"
        ]
        is True
    )


def test_repeat_series_has_unique_space_and_cli_reconstructs_without_inference(
    local, tmp_path, monkeypatch, capsys
):
    lib, kwargs, _ = local
    config = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    args = [
        "alert-series",
        "--manifest",
        str(MANIFEST),
        "--profile",
        str(config),
        "--artifacts-dir",
        str(tmp_path / "runs"),
        "--taxonomy",
        str(TAXONOMY),
    ]
    assert main(args) == 0
    first_output = json.loads(capsys.readouterr().out)
    first = tmp_path / "runs" / first_output["series_id"]
    old = {p: p.read_bytes() for p in first.rglob("*") if p.is_file()}
    assert main(args) == 0
    second_output = json.loads(capsys.readouterr().out)
    assert second_output["series_id"] != first_output["series_id"]
    assert all(p.read_bytes() == data for p, data in old.items())
    calls = len(lib.requests)
    assert main(["alert-report", "--index", str(first / "index.json")]) == 0
    assert json.loads(capsys.readouterr().out) == first_output
    relocated = tmp_path / "relocated-series"
    first.rename(relocated)
    assert main(["alert-report", "--index", str(relocated / "index.json")]) == 0
    assert json.loads(capsys.readouterr().out) == first_output
    index = json.loads((relocated / "index.json").read_text())
    for entry in index["entries"]:
        for name, link in entry["artifacts"].items():
            assert link is not None
            if name == "directory":
                assert (relocated / link).is_dir()
            else:
                assert (relocated / link).is_file()
    assert len(lib.requests) == calls


@pytest.mark.parametrize(
    ("filename", "interrupt"),
    [
        ("result.txt.tmp", False),
        ("metadata.json.tmp", False),
        ("metadata.json.tmp", True),
    ],
)
def test_run_storage_failure_is_explicit_and_preserves_actual_completed_outcome(
    local, tmp_path, monkeypatch, filename, interrupt
):
    lib, kwargs, _ = local
    profile = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    users = []
    factory, write = RoleProfile.create_experiment, Path.write_text

    def create(profile, base, directory):
        if users:
            index = json.loads((directory.parent / "index.json").read_text())
            assert index["entries"][0]["storage_error"]["phase"] == "save"
            assert users[-1].wait(0)
        user = factory(profile, base, directory)
        users.append(user)
        return user

    def persist(path, *args, **kwargs):
        if users and path.parent.name == users[0].status().run_id:
            if path.name == filename:
                assert users[0].wait(0)
                assert users[0].status().state == "COMPLETED"
                if interrupt:
                    signal.raise_signal(signal.SIGINT)
                raise OSError("cannot write final artifact")
        return write(path, *args, **kwargs)

    monkeypatch.setattr(RoleProfile, "create_experiment", create)
    monkeypatch.setattr(Path, "write_text", persist)
    result = run_alert_series(
        MANIFEST, profile, tmp_path / "runs", taxonomy_path=TAXONOMY
    )
    assert result.exit_code == (130 if interrupt else 1)
    index = json.loads((result.directory / "index.json").read_text())
    first = index["entries"][0]
    assert first["state"] == "COMPLETED" and first["error"] is None
    assert first["storage_error"] == {
        "phase": "save",
        "type": "OSError",
        "message": "cannot write final artifact",
    }
    assert first["evaluation"]["success"] is True
    assert first["classification"] == {
        "evaluated": False,
        "success": None,
        "reason": "storage_error",
    }
    assert all(u.status().state == "COMPLETED" and u.wait(0) for u in users)
    assert len(users) == len(index["entries"]) == (1 if interrupt else 32)
    assert index["interrupted"] is interrupt
    assert result.report["complete"] is (not interrupt)
    assert (
        sum(g["evaluated"] for g in result.report["groups"].values()) == len(users) - 1
    )
    assert reconstruct_report(result.directory / "index.json") == result.report


@pytest.mark.parametrize(
    "filename", ["index.json", "report.json", "final-settings.json"]
)
def test_shared_series_write_failure_terminates_command_after_join(
    local, tmp_path, monkeypatch, capsys, filename
):
    import tai.series as series

    lib, kwargs, _ = local
    profile = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    users = []
    factory, write = RoleProfile.create_experiment, series._write_json

    def create(*args):
        user = factory(*args)
        users.append(user)
        return user

    def persist(path, value):
        if users and path.name == filename:
            assert users[-1].wait(0)
            raise OSError("shared series storage unavailable")
        return write(path, value)

    monkeypatch.setattr(RoleProfile, "create_experiment", create)
    monkeypatch.setattr(series, "_write_json", persist)
    assert (
        main(
            [
                "alert-series",
                "--manifest",
                str(MANIFEST),
                "--profile",
                str(profile),
                "--artifacts-dir",
                str(tmp_path / "runs"),
                "--taxonomy",
                str(TAXONOMY),
            ]
        )
        == 2
    )
    assert "shared series storage unavailable" in capsys.readouterr().err
    assert len(users) == (32 if filename == "final-settings.json" else 1)
    assert all(user.wait(0) for user in users)


@pytest.mark.parametrize(
    ("unavailable", "interrupt"),
    [
        (("result.txt",), False),
        (("events.jsonl",), False),
        (("metadata.json",), False),
        (("result.txt", "events.jsonl", "metadata.json"), False),
        (("result.txt",), True),
    ],
)
def test_artifact_inspection_error_preserves_attempt_and_continues_real_fitd(
    local, tmp_path, monkeypatch, unavailable, interrupt
):
    lib, kwargs, _ = local
    config = qwen_profile(tmp_path, kwargs)
    positive_responses(lib, monkeypatch)
    manifest = load_alert_manifest(MANIFEST, taxonomy_path=TAXONOMY)
    factory, is_file = RoleProfile.create_experiment, Path.is_file
    users, failed_stats = [], []

    def create(profile, base, artifacts_dir):
        assert base.text == manifest.records[len(users)].base_request
        if users:
            assert users[-1].wait(0), "next attempt overlapped finalization"
            saved = json.loads((artifacts_dir.parent / "index.json").read_text())
            assert len(saved["entries"]) == len(users)
            assert saved["entries"][0]["run_id"] == users[0].status().run_id
        user = factory(profile, base, artifacts_dir)
        users.append(user)
        return user

    def inspect(path):
        if (
            path.name in unavailable
            and users
            and path.parent.name == users[0].status().run_id
        ):
            assert users[0].wait(0), "fault must occur after the worker is joined"
            assert users[0].status().state == "COMPLETED"
            failed_stats.append(path.name)
            if interrupt:
                signal.raise_signal(signal.SIGINT)
            raise OSError(errno.EIO, "artifact stat failed")
        return is_file(path)

    monkeypatch.setattr(RoleProfile, "create_experiment", create)
    monkeypatch.setattr(Path, "is_file", inspect)
    previous_handler = signal.getsignal(signal.SIGINT)
    result = run_alert_series(
        MANIFEST, config, tmp_path / "runs", taxonomy_path=TAXONOMY
    )
    assert signal.getsignal(signal.SIGINT) is previous_handler
    assert failed_stats == list(unavailable)
    assert result.exit_code == (130 if interrupt else 1)
    assert result.report["complete"] is (not interrupt)
    index = json.loads((result.directory / "index.json").read_text())
    expected_attempts = 1 if interrupt else 32
    assert len(index["entries"]) == len(users) == expected_attempts
    assert [entry["id"] for entry in index["entries"]] == [
        r.id for r in manifest.records[:expected_attempts]
    ]
    assert all(user.wait(0) for user in users)
    for entry in index["entries"]:
        run_dir = result.directory / "runs" / entry["run_id"]
        metadata = json.loads((run_dir / "metadata.json").read_text())
        for key in ("state", "reason", "run_id", "progress", "result_id", "evaluation"):
            assert entry[key] == metadata[key]
        assert entry["state"] == "COMPLETED" and entry["error"] is None
        assert entry["result"] == {"kind": "Message", "id": metadata["result"]["id"]}
        assert entry["evaluation"]["success"] is True
    affected = index["entries"][0]
    assert affected["artifact_errors"] == {
        name: {"type": "OSError", "message": "[Errno 5] artifact stat failed"}
        for name in unavailable
    }
    for name in ("result.txt", "events.jsonl", "metadata.json"):
        assert affected["artifacts"][name] == (
            None if name in unavailable else f"runs/{affected['run_id']}/{name}"
        )
    assert affected["classification"] == {
        "evaluated": False,
        "success": None,
        "reason": "artifact_inspection_error",
    }
    assert all(not entry["artifact_errors"] for entry in index["entries"][1:])
    assert all(entry["classification"]["evaluated"] for entry in index["entries"][1:])
    groups = result.report["groups"].values()
    assert sum(g["total"] for g in groups) == expected_attempts
    assert sum(g["evaluated"] for g in groups) == expected_attempts - 1
    assert sum(g["successful"] for g in groups) == expected_attempts - 1
    assert sum(g["unevaluated"] for g in groups) == 1
    calls = len(lib.requests)
    assert reconstruct_report(result.directory / "index.json") == result.report
    assert json.loads((result.directory / "report.json").read_text()) == result.report
    assert len(lib.requests) == calls


@pytest.mark.parametrize(
    "problem",
    ["last_record", "profile", "rounds", "evaluator", "taxonomy", "artifacts"],
)
def test_all_inputs_are_validated_before_any_start(
    local, tmp_path, monkeypatch, problem
):
    lib, kwargs, _ = local
    profile = qwen_profile(tmp_path, kwargs)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(MANIFEST.read_bytes())
    taxonomy = TAXONOMY
    artifacts = tmp_path / "runs"
    if problem == "last_record":
        data = json.loads(manifest.read_text())
        data["records"][-1]["base_request"] = "Changed"
        manifest.write_text(json.dumps(data))
    elif problem == "profile":
        profile.write_text(
            profile.read_text().replace('system = "Evaluator"', "unknown = true")
        )
    elif problem == "rounds":
        profile.write_text(
            profile.read_text().replace(
                "max_calls_per_conv = 20", "max_calls_per_conv = 19"
            )
        )
    elif problem == "evaluator":
        profile.write_bytes(Path("examples/fitd-qwen.toml").read_bytes())
    elif problem == "taxonomy":
        taxonomy = tmp_path / "absent.md"
    else:
        artifacts.write_text("not a directory")
    monkeypatch.setattr(
        FITDUser, "start", lambda user: pytest.fail("start on invalid input")
    )
    assert (
        main(
            [
                "alert-series",
                "--manifest",
                str(manifest),
                "--profile",
                str(profile),
                "--artifacts-dir",
                str(artifacts),
                "--taxonomy",
                str(taxonomy),
            ]
        )
        == 2
    )
    assert lib.requests == [] and lib.loads == []
    if problem != "artifacts":
        assert not artifacts.exists()
