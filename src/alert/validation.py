"""Pure validation of the complete, pinned ALERT manifest contract."""

from .records import (
    BASE_REPRESENTATION, EXTENDED_ALGORITHM, EXTENDED_COUNT, LEGACY_SELECTED_IDS,
    MAX_SEED, RECORD_ORDER, SELECTION_ALGORITHM, SOURCE_COUNT,
    AlertManifest, AlertRecord, check_coverage, source_metadata, source_record, taxonomy_metadata,
)
from .taxonomy import CATEGORY_ORDER
from .values import array_value, expect, integer_value, object_value

EXTENDED_SCHEMA_VERSION = 2


def _selection(value: object, extended: bool) -> dict[str, object]:
    fields = {"seed", "algorithm", "record_order", "category_order", "selected_ids"}
    if extended:
        fields |= {"per_category", "legacy_selected_ids"}
    selection = object_value(value, "selection", fields)
    expect(selection["algorithm"], EXTENDED_ALGORITHM if extended else SELECTION_ALGORITHM, "selection.algorithm")
    expect(selection["record_order"], RECORD_ORDER, "selection.record_order")
    expect(selection["category_order"], list(CATEGORY_ORDER), "selection.category_order")
    if extended:
        expect(selection["per_category"], EXTENDED_COUNT, "selection.per_category")
        expect(selection["legacy_selected_ids"], list(LEGACY_SELECTED_IDS), "selection.legacy_selected_ids")
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
    for index, row in enumerate(array_value(value, "records"), 1):
        record = _record(row, index)
        if record.id in identifiers:
            raise ValueError(f"Duplicate selected ALERT id: {record.id}")
        identifiers.add(record.id)
        records.append(record)
    check_coverage(records)
    return tuple(records)


def _check_extended_records(records: tuple[AlertRecord, ...]) -> None:
    expect(len(records), len(CATEGORY_ORDER) * EXTENDED_COUNT, "record count")
    expected_order = [category for category in CATEGORY_ORDER for _ in range(EXTENDED_COUNT)]
    expect([r.micro_category for r in records], expected_order, "records order")
    for index, identifier in enumerate(LEGACY_SELECTED_IDS):
        group = records[index * EXTENDED_COUNT:(index + 1) * EXTENDED_COUNT]
        if identifier not in {record.id for record in group}:
            raise ValueError(f"Missing legacy ALERT id: {identifier}")


def _check_legacy_records(records: tuple[AlertRecord, ...]) -> None:
    categories = [record.micro_category for record in records]
    if len(categories) != len(set(categories)):
        duplicate = next(category for category in categories if categories.count(category) > 1)
        raise ValueError(f"Duplicate selected ALERT micro category: {duplicate}")
    expect(len(records), len(CATEGORY_ORDER), "record count")
    expect(categories, list(CATEGORY_ORDER), "records order")


def validate_manifest(value: object) -> AlertManifest:
    data = object_value(value, "manifest", {
        "schema_version", "source", "taxonomy", "selection", "base_request_representation", "records",
    })
    version = data["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version not in (1, EXTENDED_SCHEMA_VERSION):
        raise ValueError("schema_version: expected 1 or 2")
    extended = version == EXTENDED_SCHEMA_VERSION
    expect(data["source"], source_metadata(), "source metadata")
    expect(data["taxonomy"], taxonomy_metadata(), "taxonomy metadata")
    expect(data["base_request_representation"], BASE_REPRESENTATION, "base representation")
    selection = _selection(data["selection"], extended)
    seed = integer_value(selection["seed"], "selection.seed", maximum=MAX_SEED)
    records = _records(data["records"])
    if extended:
        _check_extended_records(records)
    else:
        _check_legacy_records(records)
    expect(selection["selected_ids"], [r.id for r in records], "selection.selected_ids")
    return AlertManifest(seed, records, EXTENDED_COUNT if extended else 1)
