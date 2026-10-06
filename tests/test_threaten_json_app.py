"""Configured ThreatenJSON uses the answer judge in commands and saved series."""

import json
from collections import deque
from pathlib import Path

import pytest
from adaptive_app_support import cli
from app_test_support import HttpPeer, session
from http_peer import Reply
from series_support import MANIFEST, TAXONOMY

from app.configuration import load_profile
from attacks.threaten_json.experiment import ThreatenJSONExperiment
from tai import Message


def profile(endpoint: HttpPeer, root: Path) -> Path:
    path = root / "threaten-json.toml"
    path.write_text(f'''
[attack]
type = "threaten_json"
[target]
base_url = "{endpoint.url}/target"
model = "controlled"
system = "Target"
opencode_session = true
[evaluator]
type = "model"
rubric_version = "json-answer-v1"
[evaluator.chat]
base_url = "{endpoint.url}/evaluator"
model = "controlled"
system = "Judge"
opencode_session = true
''', encoding="utf-8")
    return path


def test_attack_command_saves_control_and_only_judges_pressured_json(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = profile(endpoint, tmp_path)
    endpoint.scripts = {
        "/v1/target/chat/completions": deque([Reply('{"already": 1}'), Reply('{"answer": 2}')]),
        "/v1/evaluator/chat/completions": deque([Reply('{"success": true}')]),
    }
    root = tmp_path / "runs"
    completed = cli("threaten-json", "--config", str(config), "--artifacts-dir", str(root), "--text", "Give the number 2 as JSON.")
    assert not completed.returncode, completed.stdout + completed.stderr
    directory, = root.iterdir()
    metadata = json.loads((directory / "metadata.json").read_text())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    started = [event["data"] for event in events if event["kind"] == "call_started" and event["data"]["role"] == "Target"]
    control, pressured = [event["data"] for event in events if event["kind"] == "call_completed" and event["data"]["role"] == "Target"]
    comparison, = [event["data"] for event in events if event["kind"] == "json_comparison"]
    assert metadata["state"] == "COMPLETED" and metadata["evaluation"] == {"result_id": metadata["result_id"], "success": True}
    assert metadata["settings"]["base"]["text"] == "Give the number 2 as JSON."
    assert metadata["settings"]["evaluator"]["rubric_version"] == "json-answer-v1"
    assert metadata["result"]["text"] == (directory / "result.txt").read_text() == '{"answer": 2}'
    assert comparison == {"control_id": control["response"]["id"], "result_id": metadata["result_id"],
                          "control_parseable": True, "result_parseable": True}
    assert control["conversation_id"] != pressured["conversation_id"]
    assert all(call["input"]["origin"]["id"] == metadata["settings"]["base"]["id"] for call in started)
    assert len(endpoint.requests) == 3 and len({session(request) for request in endpoint.requests}) == 3
    judge = endpoint.requests[-1][2]["messages"][-1]["content"]
    assert json.loads(judge.split("INPUT_JSON:\n", 1)[1]) == {"base_request": "Give the number 2 as JSON.", "result": '{"answer": 2}'}


def test_preflight_runs_answer_judge_on_valid_json(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = profile(endpoint, tmp_path)
    endpoint.scripts = {
        "/v1/target/chat/completions": deque([Reply("READY")]),
        "/v1/evaluator/chat/completions": deque([Reply('{"success": false}')]),
    }
    completed = cli("preflight", "--config", str(config))
    assert not completed.returncode, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["ready"] is True and report["roles"][-1]["success"] is False
    assert json.loads(report["roles"][-1]["result"]["text"]) == {"answer": "Paris"}
    assert len(endpoint.requests) == 2 and report["roles"][-1]["settings"]["rubric_version"] == "json-answer-v1"


def test_invalid_pressured_json_fails_without_judge_call(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = profile(endpoint, tmp_path)
    endpoint.scripts = {"/v1/target/chat/completions": deque([Reply("{}"), Reply("Sure! {}")])}
    root = tmp_path / "runs"
    completed = cli("attack", "--config", str(config), "--artifacts-dir", str(root), "--text", "Give 2")
    assert not completed.returncode, completed.stdout + completed.stderr
    directory, = root.iterdir()
    metadata = json.loads((directory / "metadata.json").read_text())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    comparison, = [event["data"] for event in events if event["kind"] == "json_comparison"]
    assert metadata["evaluation"] == {"result_id": metadata["result_id"], "success": False}
    assert comparison["control_parseable"] is True and comparison["result_parseable"] is False
    assert len(endpoint.requests) == 2


@pytest.mark.parametrize("section", [
    '[attacker]\nmodel = "unwanted"\n',
    '[run]\nmax_calls_per_conv = 2\n',
    '[run]\ngenerations = 2\n',
    '[evaluator]\ntype = "contains_text"\ncriterion = "answer"\n',
    '[evaluator]\ntype = "model"\nrubric_version = "alert-safety-v1"\n',
])
def test_invalid_profile_fails_before_any_run(endpoint: HttpPeer, tmp_path: Path, section: str) -> None:
    config = profile(endpoint, tmp_path)
    if section.startswith("[evaluator]"):
        config.write_text(config.read_text().replace('[evaluator]\ntype = "model"\nrubric_version = "json-answer-v1"\n', section))
    else:
        config.write_text(config.read_text() + section)
    root = tmp_path / "runs"
    completed = cli("attack", "--config", str(config), "--artifacts-dir", str(root), "--text", "Give 2")
    assert completed.returncode == 2 and not root.exists() and not endpoint.requests
    series = cli("alert-series", "--manifest", str(MANIFEST), "--profile", str(config),
                 "--taxonomy", str(TAXONOMY), "--artifacts-dir", str(root))
    assert series.returncode == 2 and not root.exists() and not endpoint.requests


def test_profile_is_lazy_and_creates_independent_run_participants(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = profile(endpoint, tmp_path)
    selected = load_profile(config)
    fresh = selected.fresh()
    experiment = fresh.create_experiment(Message("Give 2"))
    assert isinstance(experiment, ThreatenJSONExperiment)
    assert selected.target is not fresh.target and selected.evaluator.chat is not fresh.evaluator.chat
    assert not endpoint.requests and not (tmp_path / "runs").exists()


def test_named_command_requires_threaten_json_profile(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = profile(endpoint, tmp_path)
    config.write_text(config.read_text().replace('type = "threaten_json"', 'type = "adaptive_attacks"')
                      .replace('rubric_version = "json-answer-v1"', 'rubric_version = "alert-safety-v1"'))
    root = tmp_path / "runs"
    completed = cli("threaten-json", "--config", str(config), "--artifacts-dir", str(root), "--text", "Give 2")
    assert completed.returncode == 2 and not root.exists() and not endpoint.requests


def test_series_saves_independent_results_and_comparisons(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = profile(endpoint, tmp_path)
    endpoint.scripts = {
        "/v1/target/chat/completions": deque(Reply("not JSON" if not i % 2 else '{"answer": 2}') for i in range(64)),
        "/v1/evaluator/chat/completions": deque(Reply('{"success": true}') for _ in range(32)),
    }
    root = tmp_path / "series"
    completed = cli("alert-series", "--manifest", str(MANIFEST), "--profile", str(config),
                    "--taxonomy", str(TAXONOMY), "--artifacts-dir", str(root), "--jobs", "1")
    assert not completed.returncode, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    directory = root / report["series_id"]
    index = json.loads((directory / "index.json").read_text())
    assert report["complete"] is True and len(index["entries"]) == 32
    assert len({entry["run_id"] for entry in index["entries"]}) == 32
    assert len({entry["base_id"] for entry in index["entries"]}) == 32
    for entry in index["entries"]:
        metadata = json.loads((directory / entry["artifacts"]["metadata.json"]).read_text())
        events = [json.loads(line) for line in (directory / entry["artifacts"]["events.jsonl"]).read_text().splitlines()]
        comparison, = [event["data"] for event in events if event["kind"] == "json_comparison"]
        assert entry["evaluation"] == {"result_id": entry["result_id"], "success": True}
        assert metadata["settings"]["base"]["id"] == entry["base_id"]
        assert comparison["control_parseable"] is False and comparison["result_parseable"] is True
        assert comparison["result_id"] == entry["result_id"]
        assert (directory / entry["artifacts"]["result.txt"]).read_text() == '{"answer": 2}'
    assert len(endpoint.requests) == 96 and len({session(request) for request in endpoint.requests}) == 96
