"""Offline ALERT manifest contract. Loading never downloads data or invokes models."""

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tai._alert_taxonomy import (
    ALIASES_VERSION,
    CATEGORY_ALIASES,
    CATEGORY_ORDER,
    MACRO_CATEGORIES,
    MICRO_TO_MACRO,
    PAPER_URL,
    TAXONOMY_SHA256,
    TAXONOMY_VERSION,
    canonical_category,
)

SOURCE_REVISION = "c76d0637f4573742982d5fe961a31bdb3d1e2c80"
SOURCE_URL = (
    f"https://huggingface.co/datasets/Babelscape/ALERT/resolve/{SOURCE_REVISION}"
    "/alert.jsonl"
)
SOURCE_SHA256 = "0a5594d47d3c4b81462048e6e5dfa22304be6eb490019ddbddc73adc4e317cc4"
SOURCE_SIZE = 2_216_517
SOURCE_COUNT = 14_763
DEFAULT_SEED = 20260915
SELECTION_ALGORITHM = "python3.12-random-mt19937-choice-v1"
RECORD_ORDER = "source id ascending (numeric), within each canonical micro category"
BASE_REPRESENTATION = "raw-prompt-v1"


def _source_metadata() -> dict[str, Any]:
    return {
        "dataset": "Babelscape/ALERT",
        "config": "alert",
        "split": "test",
        "revision": SOURCE_REVISION,
        "file": "alert.jsonl",
        "url": SOURCE_URL,
        "sha256": SOURCE_SHA256,
        "size_bytes": SOURCE_SIZE,
        "record_count": SOURCE_COUNT,
        "license": "cc-by-nc-sa-4.0",
    }


def _taxonomy_metadata() -> dict[str, Any]:
    return {
        "version": TAXONOMY_VERSION,
        "document": "ALERT_taxonomy.md",
        "sha256": TAXONOMY_SHA256,
        "source": PAPER_URL,
        "location": "Section 3, Figure 2",
        "macro_categories": {
            key: list(value) for key, value in MACRO_CATEGORIES.items()
        },
        "aliases_version": ALIASES_VERSION,
        "aliases": dict(CATEGORY_ALIASES),
    }


@dataclass(frozen=True)
class AlertRecord:
    """One selected source row and its unchanged base request."""

    id: int
    prompt: str
    source_category: str
    micro_category: str
    macro_category: str
    base_request: str

    def dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlertManifest:
    """Validated by load_alert_manifest; records are in experiment input order."""

    seed: int
    records: tuple[AlertRecord, ...]

    def dump(self) -> dict[str, Any]:
        """A fresh JSON-compatible snapshot including all provenance metadata."""
        return {
            "schema_version": 1,
            "source": _source_metadata(),
            "taxonomy": _taxonomy_metadata(),
            "selection": {
                "seed": self.seed,
                "algorithm": SELECTION_ALGORITHM,
                "record_order": RECORD_ORDER,
                "category_order": list(CATEGORY_ORDER),
                "selected_ids": [record.id for record in self.records],
            },
            "base_request_representation": BASE_REPRESENTATION,
            "records": [record.dump() for record in self.records],
        }


def _object(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected a JSON object")
    missing, extra = fields - value.keys(), value.keys() - fields
    if missing or extra:
        raise ValueError(
            f"{name}: missing fields {sorted(missing)}; unknown fields {sorted(extra)}"
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected nonempty text")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ValueError(f"{name}: expected valid Unicode text") from None
    return value


def _integer(value: Any, name: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        bound = "" if maximum is None else f" <= {maximum}"
        raise ValueError(f"{name}: expected a nonnegative integer{bound}")
    return value


def _expect(value: Any, expected: Any, name: str) -> None:
    # JSON equality distinguishes true/1/1.0, unlike Python container equality.
    if json.dumps(value, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError(
            f"{name}: does not match the supported ALERT manifest contract"
        )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _decode_json(text: str) -> Any:
    return json.loads(
        text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
    )


def _check_taxonomy(path: Path) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != TAXONOMY_SHA256:
        raise ValueError(
            f"{path}: taxonomy SHA-256 mismatch; expected {TAXONOMY_SHA256}"
        )


def _source_record(value: Any, name: str) -> AlertRecord:
    data = _object(value, name, {"id", "prompt", "category"})
    identifier = _integer(data["id"], f"{name}.id")
    prompt = _text(data["prompt"], f"{name}.prompt")
    label = _text(data["category"], f"{name}.category")
    try:
        micro = canonical_category(label)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from None
    return AlertRecord(identifier, prompt, label, micro, MICRO_TO_MACRO[micro], prompt)


def _check_coverage(records: Iterable[AlertRecord]) -> None:
    categories = {record.micro_category for record in records}
    missing = [category for category in CATEGORY_ORDER if category not in categories]
    if missing:
        raise ValueError(f"Missing ALERT micro categories: {', '.join(missing)}")


def select_alert_records(
    rows: Iterable[object], *, seed: int = DEFAULT_SEED
) -> tuple[AlertRecord, ...]:
    """Validate every source row, then sample one per micro category (no I/O).

    This pure sampler also accepts controlled fixtures. Only preparation verifies
    the official source checksum; sampling alone does not attest provenance.
    """
    _integer(seed, "selection.seed", maximum=2**64 - 1)
    records = []
    identifiers: set[int] = set()
    for index, row in enumerate(rows, 1):
        record = _source_record(row, f"source row {index}")
        if record.id in identifiers:
            raise ValueError(f"Duplicate ALERT source id: {record.id}")
        identifiers.add(record.id)
        records.append(record)
    _check_coverage(records)
    groups: dict[str, list[AlertRecord]] = {category: [] for category in CATEGORY_ORDER}
    for record in sorted(records, key=lambda record: record.id):
        groups[record.micro_category].append(record)
    rng = random.Random()
    rng.seed(seed, version=2)
    return tuple(rng.choice(groups[category]) for category in CATEGORY_ORDER)


def load_alert_manifest(path: Path, *, taxonomy_path: Path) -> AlertManifest:
    """Load and validate the complete v1 manifest and local taxonomy before inference.

    No source download, model construction, or lazy per-row validation occurs.
    ValueError diagnoses invalid content; OSError diagnoses missing/unreadable files.
    Source membership is verified during preparation, not by this offline loader.
    """
    _check_taxonomy(taxonomy_path)
    data = _object(
        _decode_json(path.read_bytes().decode("utf-8")),
        "manifest",
        {
            "schema_version",
            "source",
            "taxonomy",
            "selection",
            "base_request_representation",
            "records",
        },
    )
    _expect(data["schema_version"], 1, "schema_version")
    _expect(data["source"], _source_metadata(), "source metadata")
    _expect(data["taxonomy"], _taxonomy_metadata(), "taxonomy metadata")
    _expect(
        data["base_request_representation"], BASE_REPRESENTATION, "base representation"
    )
    selection = _object(
        data["selection"],
        "selection",
        {"seed", "algorithm", "record_order", "category_order", "selected_ids"},
    )
    seed = _integer(selection["seed"], "selection.seed", maximum=2**64 - 1)
    _expect(selection["algorithm"], SELECTION_ALGORITHM, "selection.algorithm")
    _expect(selection["record_order"], RECORD_ORDER, "selection.record_order")
    _expect(
        selection["category_order"], list(CATEGORY_ORDER), "selection.category_order"
    )
    if not isinstance(data["records"], list):
        raise ValueError("records: expected a JSON array")
    records = []
    identifiers: set[int] = set()
    categories: set[str] = set()
    for index, value in enumerate(data["records"], 1):
        name = f"records[{index}]"
        row = _object(
            value,
            name,
            {
                "id",
                "prompt",
                "source_category",
                "micro_category",
                "macro_category",
                "base_request",
            },
        )
        record = _source_record(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "category": row["source_category"],
            },
            name,
        )
        _integer(record.id, f"{name}.id", maximum=SOURCE_COUNT - 1)
        _expect(row["micro_category"], record.micro_category, f"{name}.micro_category")
        _expect(row["macro_category"], record.macro_category, f"{name}.macro_category")
        _expect(row["base_request"], record.prompt, f"{name}.base_request")
        if record.id in identifiers:
            raise ValueError(f"Duplicate selected ALERT id: {record.id}")
        if record.micro_category in categories:
            raise ValueError(
                f"Duplicate selected ALERT micro category: {record.micro_category}"
            )
        identifiers.add(record.id)
        categories.add(record.micro_category)
        records.append(record)
    _check_coverage(records)
    _expect(len(records), 32, "record count")
    _expect([r.micro_category for r in records], list(CATEGORY_ORDER), "records order")
    _expect(
        selection["selected_ids"], [r.id for r in records], "selection.selected_ids"
    )
    return AlertManifest(seed, tuple(records))
