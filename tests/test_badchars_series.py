"""A complete offline ALERT series retains one entry per base, not per variant."""

import json
from collections import deque
from pathlib import Path

from adaptive_app_support import cli
from app_test_support import HttpPeer, session
from badchars_app_support import badchars_profile
from http_peer import Reply
from series_support import MANIFEST, TAXONOMY

from alert import load_alert_manifest
from alert.report import reconstruct_report
from attacks.badchars.garak_adapter import candidates
from attacks.badchars.profile import DeletionProfile
from tai import Message


def test_full_series_exhausts_internal_variants_and_rebuilds_saved_final_evaluations(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = badchars_profile(endpoint, tmp_path, "perturbation_budget = 2\nmax_position_candidates = 2\nmax_ascii_variants = 1")
    endpoint.scripts = {
        "/v1/target/chat/completions": deque(Reply("Complete answer") for _ in range(96)),
        "/v1/evaluator/chat/completions": deque(Reply('{"success": false}') for _ in range(96)),
    }
    root = tmp_path / "series"
    completed = cli("alert-series", "--manifest", str(MANIFEST), "--profile", str(config),
                    "--artifacts-dir", str(root), "--taxonomy", str(TAXONOMY), "--jobs", "3")
    assert not completed.returncode, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["complete"] is True
    assert "Backend: OpenAI-compatible" in completed.stderr
    directory = root / report["series_id"]
    _assert_entries(directory)
    _assert_requests(endpoint)
    effective = json.loads((directory / "effective-profile.json").read_text())
    assert effective["attack"] == {"type": "badchars_deletion"}
    assert effective["badchars_deletion"] == dict(DeletionProfile(2, 2, 1).configuration().parameters)
    assert "fitd" not in effective and "Attacker" not in effective
    assert reconstruct_report(directory / "index.json") == report
    rebuilt = cli("alert-report", "--index", str(directory / "index.json"))
    assert not rebuilt.returncode and json.loads(rebuilt.stdout) == report
    assert len(endpoint.requests) == 192


def _assert_entries(directory: Path) -> None:
    manifest = load_alert_manifest(MANIFEST, taxonomy_path=TAXONOMY)
    index = json.loads((directory / "index.json").read_text())
    assert len(index["entries"]) == 32
    assert [entry["id"] for entry in index["entries"]] == [record.id for record in manifest.records]
    assert len({entry["run_id"] for entry in index["entries"]}) == 32
    assert len({entry["base_id"] for entry in index["entries"]}) == 32
    for entry, record in zip(index["entries"], manifest.records, strict=True):
        metadata = json.loads((directory / entry["artifacts"]["metadata.json"]).read_text())
        assert metadata["settings"]["base"]["text"] == record.base_request
        assert metadata["settings"]["attack"] == {"type": "badchars_deletion"}
        assert "Attacker" not in metadata["settings"]
        assert entry["reason"] == "strategy_exhausted"
        assert entry["evaluation"] == metadata["evaluation"] == {"result_id": metadata["result_id"], "success": False}
        assert entry["progress"]["target_calls"] == entry["progress"]["evaluator_calls"] == 3
        assert entry["progress"]["max_calls_per_conv"] is None
        assert (directory / entry["artifacts"]["result.txt"]).read_text() == "Complete answer"
        _assert_events(directory / entry["artifacts"]["events.jsonl"], record.base_request, metadata["result_id"])


def _assert_events(path: Path, base: str, result_id: str) -> None:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    evaluations = [event["data"] for event in events if event["kind"] == "evaluation_completed"]
    assert evaluations[-1]["result_id"] == result_id
    started = [event["data"] for event in events if event["kind"] == "call_started" and event["data"]["role"] == "Target"]
    assert [event["input"]["text"] for event in started] == [item.request.text for item in candidates(Message(base), profile=DeletionProfile(2, 2, 1))]
    assert len({event["attempt_id"] for event in started}) == 3
    assert all(event["input"]["origin"]["text"] == base for event in started)


def _assert_requests(endpoint: HttpPeer) -> None:
    manifest = load_alert_manifest(MANIFEST, taxonomy_path=TAXONOMY)
    targets = [request for request in endpoint.requests if "/target/" in request[0]]
    assert len(targets) == 96 and len(endpoint.requests) == 192
    assert len({session(request) for request in endpoint.requests}) == 192
    assert all(len(request[2]["messages"]) == 2 for request in endpoint.requests)
    assert {request[2]["messages"][-1]["content"] for request in targets} == {
        item.request.text for record in manifest.records
        for item in candidates(Message(record.base_request), profile=DeletionProfile(2, 2, 1))
    }
