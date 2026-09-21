"""Accounting rejects stale/ambiguous evidence without consulting a participant."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tai.alert import load_alert_manifest
from tai.alert.report import (
    build_series_report,
    classify_outcome,
    reconstruct_report,
)


def saved_index():
    manifest = load_alert_manifest(
        Path("examples/alert-manifest.json"), taxonomy_path=Path("ALERT_taxonomy.md")
    )
    planned = [
        {"id": r.id, "macro_category": r.macro_category} for r in manifest.records
    ]
    return {
        "schema_version": 1,
        "series_id": "controlled-series",
        "manifest": "manifest.json",
        "planned": planned,
        "finished_at": "2026-09-15T00:00:00+00:00",
        "interrupted": False,
        "entries": [
            {
                **record,
                "sequence": sequence,
                "finished_at": "2026-09-15T00:00:00+00:00",
                "state": "COMPLETED",
                "error": None,
                "result": {"kind": "Message", "id": f"result-{sequence}"},
                "result_id": f"result-{sequence}",
                "evaluation": {"result_id": f"result-{sequence}", "success": True},
            }
            for sequence, record in enumerate(planned, 1)
        ],
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "FAILED"},
        {"state": "STOPPED"},
        {"state": "RUNNING"},
        {"state": "IDLE"},
        {"state": None},
        {"error": {"phase": "start", "message": "failure"}},
        {"result": None},
        {"result": {}},
        {"result": {"kind": "str", "id": "result-1"}},
        {"result": {"kind": "Message", "id": ""}},
        {"result_id": None},
        {"result_id": "old"},
        {"evaluation": {"result_id": "old", "success": True}},
        {"evaluation": {"result_id": "result-1", "success": True, "extra": 1}},
        *[{"evaluation": value} for value in (None, {}, [], True, "true")],
        *[
            {"evaluation": {"result_id": "result-1", "success": value}}
            for value in (0, 1, 1.0, "true", "false", None, [], {})
        ],
    ],
)
def test_only_completed_actual_final_message_with_matching_boolean_counts(changes):
    entry = saved_index()["entries"][0]
    entry.update(changes)
    entry["classification"] = {"evaluated": True, "success": True}
    assert classify_outcome(entry)["evaluated"] is False
    assert classify_outcome(entry)["success"] is None


def test_counts_weight_runs_not_categories_and_zero_evaluated_is_null(tmp_path):
    index = saved_index()
    for entry in index["entries"]:
        if entry["macro_category"] == "weapon":
            entry["state"] = "FAILED"  # Old success remains deliberately present.
    index["entries"][0]["evaluation"]["success"] = False
    index["entries"][1]["evaluation"] = None
    index["entries"][2]["result_id"] = "changed"
    report = build_series_report(index)
    assert report["complete"] is True
    assert report["groups"]["hate"] == {
        "total": 8,
        "evaluated": 6,
        "successful": 5,
        "unevaluated": 2,
        "success_percent": 100 * 5 / 6,
    }
    assert report["groups"]["weapon"] == {
        "total": 5,
        "evaluated": 0,
        "successful": 0,
        "unevaluated": 5,
        "success_percent": None,
    }
    assert {k: v["total"] for k, v in report["groups"].items()} == {
        "hate": 8,
        "crime": 8,
        "substance": 5,
        "sex": 3,
        "self_harm": 3,
        "weapon": 5,
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index))
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    assert reconstruct_report(path, first) == reconstruct_report(path, second) == report
    assert first.read_bytes() == second.read_bytes()
    assert b'"success_percent": null' in first.read_bytes()
    # No manifest, run metadata, result file, chat, or model is needed to rebuild.
    with pytest.raises(ValueError, match="differ"):
        reconstruct_report(path, path)


@pytest.mark.parametrize("attempts", [0, 1, 31, 32])
def test_partial_totals_only_count_attempts_and_always_include_all_groups(attempts):
    index = saved_index()
    index["entries"] = index["entries"][:attempts]
    index["interrupted"] = True
    index["complete"] = True  # Cached summaries cannot override the saved evidence.
    report = build_series_report(index)
    assert report["complete"] is False
    assert len(report["groups"]) == 6
    assert sum(group["total"] for group in report["groups"].values()) == attempts
    for group in report["groups"].values():
        if not group["total"]:
            assert group["success_percent"] is None


def test_reconstruction_rejects_reordered_duplicate_or_foreign_attempts():
    original = saved_index()
    for mutation in ("duplicate", "group", "order", "schema"):
        index = deepcopy(original)
        if mutation == "duplicate":
            index["entries"][1] = index["entries"][0]
        elif mutation == "group":
            index["entries"][0]["macro_category"] = "crime"
        elif mutation == "order":
            index["entries"].reverse()
        else:
            index["schema_version"] = True
        with pytest.raises(ValueError):
            build_series_report(index)
