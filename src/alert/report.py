"""Offline series accounting. This module never constructs or calls participants."""

from pathlib import Path
from typing import Any

from core.runs.storage import _write_json

from .manifest import _decode_json
from .taxonomy import MACRO_CATEGORIES


def classify_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    """Classify saved final evidence, never a cached classification or old event.

    ``result`` is the runner's compact witness of an actual final Message. Its
    text/provenance live in the run artifacts, not in the accounting index.
    """
    reason = "valid"
    result = outcome.get("result")
    evaluation = outcome.get("evaluation")
    if outcome.get("error") is not None:
        reason = "execution_error"
    elif outcome.get("storage_error") is not None:
        reason = "storage_error"
    elif outcome.get("artifact_errors"):
        reason = "artifact_inspection_error"
    elif outcome.get("state") != "COMPLETED":
        reason = "not_completed"
    elif (
        not isinstance(result, dict)
        or result.get("kind") != "Message"
        or not isinstance(result.get("id"), str)
        or not result["id"]
    ):
        reason = "missing_result"
    elif not isinstance(evaluation, dict) or evaluation.keys() != {
        "result_id",
        "success",
    }:
        reason = "missing_or_malformed_evaluation"
    elif type(evaluation["success"]) is not bool:
        reason = "non_boolean_evaluation"
    elif not (evaluation["result_id"] == result["id"] == outcome.get("result_id")):
        reason = "result_id_mismatch"
    valid = reason == "valid"
    success = evaluation["success"] if valid and isinstance(evaluation, dict) else None
    return {
        "evaluated": valid,
        "success": success,
        "reason": reason,
    }


def build_series_report(index: dict[str, Any]) -> dict[str, Any]:
    """Recompute all six groups from the attempted prefix in a v1 saved index.

    Invalid series identity/order is rejected. Missing or malformed verdicts are
    unevaluated. Cached ``classification``/``complete`` fields are not trusted.
    """
    if type(index.get("schema_version")) is not int or index["schema_version"] != 1:
        raise ValueError("Unsupported series index schema_version")
    if not isinstance(index.get("series_id"), str) or not index["series_id"]:
        raise ValueError("Series index requires series_id")
    if not isinstance(index.get("manifest"), str) or not index["manifest"]:
        raise ValueError("Series index requires a manifest link")
    planned, entries = index.get("planned"), index.get("entries")
    if not isinstance(planned, list) or len(planned) != 32:
        raise ValueError("Series index requires 32 planned records")
    if not isinstance(entries, list) or len(entries) > len(planned):
        raise ValueError("Series entries must be an attempted prefix of planned")
    identifiers: set[int] = set()
    planned_counts = dict.fromkeys(MACRO_CATEGORIES, 0)
    for record in planned:
        if (
            not isinstance(record, dict)
            or type(record.get("id")) is not int
            or record["id"] in identifiers
            or not isinstance(record.get("macro_category"), str)
            or record.get("macro_category") not in MACRO_CATEGORIES
        ):
            raise ValueError("Invalid or duplicate planned record")
        identifiers.add(record["id"])
        planned_counts[record["macro_category"]] += 1
    if planned_counts != {key: len(value) for key, value in MACRO_CATEGORIES.items()}:
        raise ValueError("Planned records must cover all six ALERT groups")
    groups: dict[str, dict[str, Any]] = {
        key: {"total": 0, "evaluated": 0, "successful": 0} for key in MACRO_CATEGORIES
    }
    for sequence, (entry, record) in enumerate(zip(entries, planned, strict=False), 1):
        if (
            not isinstance(entry, dict)
            or type(entry.get("id")) is not int
            or type(entry.get("sequence")) is not int
            or entry["sequence"] != sequence
            or any(entry.get(key) != value for key, value in record.items())
        ):
            raise ValueError("Series entries do not match planned order/identity")
        group = groups[record["macro_category"]]
        classification = classify_outcome(entry)
        group["total"] += 1
        group["evaluated"] += int(classification["evaluated"])
        group["successful"] += int(classification["success"] is True)
    for group in groups.values():
        group["unevaluated"] = group["total"] - group["evaluated"]
        group["success_percent"] = (
            100 * group["successful"] / group["evaluated"]
            if group["evaluated"]
            else None
        )
    return {
        "schema_version": 1,
        "series_id": index["series_id"],
        "manifest": index["manifest"],
        "complete": bool(
            index.get("finished_at")
            and index.get("interrupted") is False
            and len(entries) == len(planned)
            and all(entry.get("finished_at") for entry in entries)
        ),
        "groups": groups,
    }


def reconstruct_report(index_path: Path, output: Path | None = None) -> dict[str, Any]:
    """Read index.json only; optionally write a deterministic JSON report.

    The output parent must exist. The input index is never overwritten.
    """
    if output is not None and output.resolve() == index_path.resolve():
        raise ValueError("Report output must differ from the series index")
    index = _decode_json(index_path.read_bytes().decode("utf-8"))
    if not isinstance(index, dict):
        raise ValueError("Series index must be a JSON object")
    report = build_series_report(index)
    if output is not None:
        _write_json(output, report)
    return report
