"""Full offline ALERT series uses the real strategy and injected console journal."""

import json
from collections import deque
from pathlib import Path

from adaptive_app_support import adaptive_profile, cli
from app_test_support import HttpPeer, session
from http_peer import Reply
from series_support import MANIFEST, TAXONOMY

from alert import load_alert_manifest
from alert.report import reconstruct_report
from attacks.adaptive_attacks.garak_adapter import variants
from tai import Message


def test_full_installed_alert_series_and_report_without_inference(endpoint: HttpPeer, tmp_path: Path) -> None:
    config = adaptive_profile(endpoint, tmp_path)
    manifest = load_alert_manifest(MANIFEST, taxonomy_path=TAXONOMY)
    endpoint.scripts = {
        "/v1/target/chat/completions": deque(Reply("Complete answer") for _ in range(32)),
        "/v1/evaluator/chat/completions": deque(Reply('{"success": true}') for _ in range(32)),
    }
    root = tmp_path / "series"
    completed = cli("alert-series", "--manifest", str(MANIFEST), "--profile", str(config),
                    "--artifacts-dir", str(root), "--taxonomy", str(TAXONOMY), "--jobs", "3")
    assert not completed.returncode, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["complete"] is True
    assert "Backend: OpenAI-compatible" in completed.stderr and "УСПЕХ" in completed.stderr
    directory = root / report["series_id"]
    index = json.loads((directory / "index.json").read_text())
    assert len(index["entries"]) == 32
    assert [entry["id"] for entry in index["entries"]] == [record.id for record in manifest.records]
    assert len({entry["run_id"] for entry in index["entries"]}) == 32
    assert len({entry["base_id"] for entry in index["entries"]}) == 32
    for entry, record in zip(index["entries"], manifest.records, strict=True):
        metadata = json.loads((directory / entry["artifacts"]["metadata.json"]).read_text())
        assert metadata["settings"]["base"]["text"] == record.base_request
        assert metadata["settings"]["attack"] == {"type": "adaptive_attacks"}
        assert "Attacker" not in metadata["settings"]
        assert entry["reason"] == "evaluator_success"
        assert entry["evaluation"] == metadata["evaluation"] == {"result_id": metadata["result_id"], "success": True}
        assert (directory / entry["artifacts"]["result.txt"]).read_text() == "Complete answer"
        assert entry["progress"]["target_calls"] == 1 and entry["progress"]["max_calls_per_conv"] is None
    _assert_series_requests(endpoint, directory)
    assert reconstruct_report(directory / "index.json") == report
    rebuilt = cli("alert-report", "--index", str(directory / "index.json"))
    assert not rebuilt.returncode and json.loads(rebuilt.stdout) == report
    assert len(endpoint.requests) == 64


def _assert_series_requests(endpoint: HttpPeer, directory: Path) -> None:
    manifest = load_alert_manifest(MANIFEST, taxonomy_path=TAXONOMY)
    targets = [request for request in endpoint.requests if "/target/" in request[0]]
    assert len(targets) == 32 and len(endpoint.requests) == 64
    assert len({session(request) for request in endpoint.requests}) == 64
    assert all(len(request[2]["messages"]) == 2 for request in endpoint.requests)
    assert {request[2]["messages"][-1]["content"] for request in targets} == {
        variants(Message(record.base_request))[0].text for record in manifest.records
    }
    effective = json.loads((directory / "effective-profile.json").read_text())
    assert effective["attack"] == {"type": "adaptive_attacks"}
    assert effective["adaptive_attacks"]["variants"] == 5
    assert "fitd" not in effective and "Attacker" not in effective
