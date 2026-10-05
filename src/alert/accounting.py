"""Pure accounting from saved final evidence and the attempted series prefix."""

from collections.abc import Mapping
from typing import TypedDict, cast

from .records import EXTENDED_COUNT, LEGACY_SELECTED_IDS, SOURCE_COUNT
from .taxonomy import CATEGORY_ORDER, MACRO_CATEGORIES, MICRO_TO_MACRO, canonical_category
from .values import array_value, object_value

COMPLETED = "COMPLETED"
MESSAGE_KIND = "Message"
VALID = "valid"
ACTIVE = "active"
FINISHED = "finished"
PARALLEL_SCHEMA = 2


class Classification(TypedDict):
    evaluated: bool
    success: bool | None
    reason: str


class GroupCounts(TypedDict):
    total: int
    evaluated: int
    successful: int
    unevaluated: int
    success_percent: float | None


class MicroCounts(TypedDict):
    attempted: int
    evaluated: int
    successful: int
    unevaluated: int
    success_percent: float | None


class SeriesReport(TypedDict):
    schema_version: int
    series_id: str
    manifest: str
    complete: bool
    groups: dict[str, GroupCounts]
    micro_categories: dict[str, MicroCounts]


def _result_id(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    result = cast(dict[str, object], value)
    identifier = result.get("id")
    if result.get("kind") == MESSAGE_KIND and isinstance(identifier, str) and identifier:
        return identifier
    return None


def _evidence(outcome: Mapping[str, object]) -> Classification:
    identifier = _result_id(outcome.get("result"))
    evaluation = outcome.get("evaluation")
    if identifier is None:
        return {"evaluated": False, "success": None, "reason": "missing_result"}
    if not isinstance(evaluation, dict) or evaluation.keys() != {"result_id", "success"}:
        return {"evaluated": False, "success": None, "reason": "missing_or_malformed_evaluation"}
    verdict = cast(dict[str, object], evaluation)
    success = verdict["success"]
    if not isinstance(success, bool):
        return {"evaluated": False, "success": None, "reason": "non_boolean_evaluation"}
    valid = verdict["result_id"] == identifier == outcome.get("result_id")
    return {"evaluated": valid, "success": success if valid else None,
            "reason": VALID if valid else "result_id_mismatch"}


def classify_outcome(outcome: Mapping[str, object]) -> Classification:
    """Classify saved final evidence, never a cached classification or old event.

    ``result`` is the runner's compact witness of an actual final Message. Its
    text/provenance live in the run artifacts, not in the accounting index.

    Args:
        outcome (Mapping[str, object]): Saved terminal evidence, possibly malformed.

    Returns:
        Classification: Whether a matching final boolean verdict is countable.
    """
    invalid = (
        (outcome.get("phase") == ACTIVE, ACTIVE),
        (outcome.get("error") is not None, "execution_error"),
        (outcome.get("storage_error") is not None, "storage_error"),
        (bool(outcome.get("artifact_errors")), "artifact_inspection_error"),
        (outcome.get("state") != COMPLETED, "not_completed"),
    )
    if (reason := next((reason for failed, reason in invalid if failed), None)) is not None:
        return {"evaluated": False, "success": None, "reason": reason}
    return _evidence(outcome)


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _identity(index: Mapping[str, object]) -> tuple[str, str]:
    if not _integer(index.get("schema_version")) or index["schema_version"] not in {1, 2}:
        raise ValueError("Unsupported series index schema_version")
    identifier, manifest = index.get("series_id"), index.get("manifest")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Series index requires series_id")
    if not isinstance(manifest, str) or not manifest:
        raise ValueError("Series index requires a manifest link")
    return identifier, manifest


def _planned(value: object) -> list[dict[str, object]]:
    planned = [object_value(row, "planned record") for row in array_value(value, "planned")]
    if len(planned) == len(CATEGORY_ORDER) * EXTENDED_COUNT:
        _ = _planned_counts(planned)
        _validate_extended_plan(planned)
    elif len(planned) != len(CATEGORY_ORDER):
        raise ValueError("Series index requires 32 or 640 planned records")
    elif _planned_counts(planned) != {key: len(value) for key, value in MACRO_CATEGORIES.items()}:
        raise ValueError("Planned records must cover all six ALERT groups")
    else:
        _validate_legacy_plan(planned)
    return planned


def _validate_legacy_plan(planned: list[dict[str, object]]) -> None:
    for category, record in zip(CATEGORY_ORDER, planned, strict=True):
        if (record.get("micro_category", category) != category
                or record["macro_category"] != MICRO_TO_MACRO[category]):
            raise ValueError("Legacy series plan disagrees with ALERT taxonomy order")


def _validate_extended_plan(planned: list[dict[str, object]]) -> None:
    for position, record in enumerate(planned):
        category = CATEGORY_ORDER[position // EXTENDED_COUNT]
        source = record.get("source_category")
        identifier = record["id"]
        try:
            source_category = canonical_category(source) if isinstance(source, str) else None
        except ValueError:
            source_category = None
        valid_id = _integer(identifier) and 0 <= cast(int, identifier) < SOURCE_COUNT
        if (record.get("micro_category") != category or source_category != category
                or record.get("macro_category") != MICRO_TO_MACRO[category] or not valid_id):
            raise ValueError("Extended series plan disagrees with ALERT selection")
        if not position % EXTENDED_COUNT and identifier != LEGACY_SELECTED_IDS[position // EXTENDED_COUNT]:
            raise ValueError("Extended series plan must start each category with its legacy selection")


def _planned_counts(planned: list[dict[str, object]]) -> dict[str, int]:
    identifiers: set[object] = set()
    counts = dict.fromkeys(MACRO_CATEGORIES, 0)
    for record in planned:
        identifier, macro = record.get("id"), record.get("macro_category")
        if not _integer(identifier) or identifier in identifiers:
            raise ValueError("Invalid or duplicate planned record")
        if not isinstance(macro, str) or macro not in MACRO_CATEGORIES:
            raise ValueError("Invalid or duplicate planned record")
        identifiers.add(identifier)
        counts[macro] += 1
    return counts


def _entries(value: object, planned: list[dict[str, object]]) -> list[dict[str, object]]:
    entries = [object_value(row, "entry") for row in array_value(value, "entries")]
    if len(entries) > len(planned):
        raise ValueError("Series entries must be an attempted prefix of planned")
    for sequence, (entry, record) in enumerate(zip(entries, planned, strict=False), 1):
        if not _integer(entry.get("id")) or not _integer(entry.get("sequence")):
            raise ValueError("Series entries do not match planned order/identity")
        if entry["sequence"] != sequence or any(entry.get(key) != value for key, value in record.items()):
            raise ValueError("Series entries do not match planned order/identity")
    return entries


def _groups(
    entries: list[dict[str, object]], planned: list[dict[str, object]],
) -> tuple[dict[str, GroupCounts], dict[str, MicroCounts]]:
    groups: dict[str, GroupCounts] = {
        key: {"total": 0, "evaluated": 0, "successful": 0, "unevaluated": 0, "success_percent": None}
        for key in MACRO_CATEGORIES
    }
    micro_categories: dict[str, MicroCounts] = {
        key: {"attempted": 0, "evaluated": 0, "successful": 0, "unevaluated": 0, "success_percent": None}
        for key in CATEGORY_ORDER
    }
    for position, entry in enumerate(entries):
        group = groups[cast(str, entry["macro_category"])]
        category = planned[position].get("micro_category")
        micro = micro_categories[cast(str, category if category is not None else CATEGORY_ORDER[position])]
        classification = classify_outcome(entry)
        group["total"] += 1
        group["evaluated"] += int(classification["evaluated"])
        group["successful"] += int(classification["success"] is True)
        micro["attempted"] += 1
        micro["evaluated"] += int(classification["evaluated"])
        micro["successful"] += int(classification["success"] is True)
    for group in groups.values():
        group["unevaluated"] = group["total"] - group["evaluated"]
        group["success_percent"] = 100 * group["successful"] / group["evaluated"] if group["evaluated"] else None
    for micro in micro_categories.values():
        micro["unevaluated"] = micro["attempted"] - micro["evaluated"]
        micro["success_percent"] = 100 * micro["successful"] / micro["evaluated"] if micro["evaluated"] else None
    return groups, micro_categories


def _validate_phases(entries: list[dict[str, object]]) -> None:
    for entry in entries:
        if entry.get("phase") not in {ACTIVE, FINISHED}:
            raise ValueError("Series v2 entry requires active/finished phase")
        if (entry["phase"] == FINISHED) != bool(entry.get("finished_at")):
            raise ValueError("Series entry phase disagrees with finished_at")
        if entry["phase"] == ACTIVE and any(entry.get(key) is not None for key in ("result", "evaluation", "result_id")):
            raise ValueError("Active entries cannot publish final evidence")


def build_series_report(index: Mapping[str, object]) -> SeriesReport:
    """Recompute six groups and 32 micro-categories from a v1 or v2 index.

    Invalid series identity/order is rejected. Missing or malformed verdicts are
    unevaluated. Cached ``classification``/``complete`` fields are not trusted.

    Args:
        index (Mapping[str, object]): Decoded saved index.

    Returns:
        SeriesReport: Recomputed counts and completeness.
    """
    identifier, manifest = _identity(index)
    planned = _planned(index.get("planned"))
    entries = _entries(index.get("entries"), planned)
    if index["schema_version"] == PARALLEL_SCHEMA:
        _validate_phases(entries)
    groups, micro_categories = _groups(entries, planned)
    return {
        "schema_version": 2, "series_id": identifier, "manifest": manifest,
        "complete": bool(index.get("finished_at") and index.get("interrupted") is False
                         and len(entries) == len(planned) and all(entry.get("finished_at") for entry in entries)),
        "groups": groups, "micro_categories": micro_categories,
    }
