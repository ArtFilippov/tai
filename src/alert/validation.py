"""Pure validation of the complete, pinned ALERT manifest contract."""

from .records import (
    BASE_REPRESENTATION, MAX_SEED, RECORD_ORDER, SELECTION_ALGORITHM, SOURCE_COUNT,
    AlertManifest, AlertRecord, check_coverage, source_metadata, source_record, taxonomy_metadata,
)
from .taxonomy import CATEGORY_ORDER
from .values import array_value, expect, integer_value, object_value


def _selection(value: object) -> dict[str, object]:
    selection = object_value(value, "selection", {"seed", "algorithm", "record_order", "category_order", "selected_ids"})
    expect(selection["algorithm"], SELECTION_ALGORITHM, "selection.algorithm")
    expect(selection["record_order"], RECORD_ORDER, "selection.record_order")
    expect(selection["category_order"], list(CATEGORY_ORDER), "selection.category_order")
    return selection


def _record(value: object, index: int) -> AlertRecord:
    name = f"records[{index}]"
    row = object_value(value, name, {"id", "prompt", "source_category", "micro_category", "macro_category", "base_request"})
    record = source_record({"id": row["id"], "prompt": row["prompt"], "category": row["source_category"]}, name)
    _ = integer_value(record.id, f"{name}.id", maximum=SOURCE_COUNT - 1)
    expect(row["micro_category"], record.micro_category, f"{name}.micro_category")
    expect(row["macro_category"], record.macro_category, f"{name}.macro_category")
    expect(row["base_request"], record.prompt, f"{name}.base_request")
    return record


def _records(value: object) -> tuple[AlertRecord, ...]:
    records: list[AlertRecord] = []
    identifiers: set[int] = set()
    categories: set[str] = set()
    for index, row in enumerate(array_value(value, "records"), 1):
        record = _record(row, index)
        if record.id in identifiers:
            raise ValueError(f"Duplicate selected ALERT id: {record.id}")
        if record.micro_category in categories:
            raise ValueError(f"Duplicate selected ALERT micro category: {record.micro_category}")
        identifiers.add(record.id)
        categories.add(record.micro_category)
        records.append(record)
    check_coverage(records)
    expect(len(records), len(CATEGORY_ORDER), "record count")
    expect([r.micro_category for r in records], list(CATEGORY_ORDER), "records order")
    return tuple(records)


def validate_manifest(value: object) -> AlertManifest:
    data = object_value(value, "manifest", {
        "schema_version", "source", "taxonomy", "selection", "base_request_representation", "records",
    })
    expect(data["schema_version"], 1, "schema_version")
    expect(data["source"], source_metadata(), "source metadata")
    expect(data["taxonomy"], taxonomy_metadata(), "taxonomy metadata")
    expect(data["base_request_representation"], BASE_REPRESENTATION, "base representation")
    selection = _selection(data["selection"])
    seed = integer_value(selection["seed"], "selection.seed", maximum=MAX_SEED)
    records = _records(data["records"])
    expect(selection["selected_ids"], [r.id for r in records], "selection.selected_ids")
    return AlertManifest(seed, records)
