"""Pure ALERT provenance values and deterministic source sampling."""

import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .taxonomy import (
    ALIASES_VERSION, CATEGORY_ALIASES, CATEGORY_ORDER, MACRO_CATEGORIES,
    MICRO_TO_MACRO, PAPER_URL, TAXONOMY_SHA256, TAXONOMY_VERSION, canonical_category,
)
from .values import JsonObject, integer_value, object_value, text_value

SOURCE_REVISION = "c76d0637f4573742982d5fe961a31bdb3d1e2c80"
SOURCE_URL = f"https://huggingface.co/datasets/Babelscape/ALERT/resolve/{SOURCE_REVISION}/alert.jsonl"
SOURCE_SHA256 = "0a5594d47d3c4b81462048e6e5dfa22304be6eb490019ddbddc73adc4e317cc4"
SOURCE_SIZE = 2_216_517
SOURCE_COUNT = 14_763
DEFAULT_SEED = 20260915
SELECTION_ALGORITHM = "python3.12-random-mt19937-choice-v1"
RECORD_ORDER = "source id ascending (numeric), within each canonical micro category"
BASE_REPRESENTATION = "raw-prompt-v1"
MAX_SEED = 2**64 - 1


def source_metadata() -> JsonObject:
    return {
        "dataset": "Babelscape/ALERT", "config": "alert", "split": "test",
        "revision": SOURCE_REVISION, "file": "alert.jsonl", "url": SOURCE_URL,
        "sha256": SOURCE_SHA256, "size_bytes": SOURCE_SIZE,
        "record_count": SOURCE_COUNT, "license": "cc-by-nc-sa-4.0",
    }


def taxonomy_metadata() -> JsonObject:
    return {
        "version": TAXONOMY_VERSION, "document": "ALERT_taxonomy.md",
        "sha256": TAXONOMY_SHA256, "source": PAPER_URL, "location": "Section 3, Figure 2",
        "macro_categories": {key: list(value) for key, value in MACRO_CATEGORIES.items()},
        "aliases_version": ALIASES_VERSION, "aliases": dict(CATEGORY_ALIASES),
    }


@dataclass(frozen=True)
class AlertRecord:
    """One selected source row and its unchanged base request.

    Attributes:
        id (int): Original numeric source identifier.
        prompt (str): Unchanged source prompt.
        source_category (str): Original source label.
        micro_category (str): Canonical micro category.
        macro_category (str): Canonical group.
        base_request (str): Text presented as the base request.
    """

    id: int
    prompt: str
    source_category: str
    micro_category: str
    macro_category: str
    base_request: str

    def dump(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class AlertManifest:
    """Validated by load_alert_manifest; records are in experiment input order.

    Attributes:
        seed (int): Deterministic sampler seed.
        records (tuple[AlertRecord, ...]): Validated selected source rows.
    """

    seed: int
    records: tuple[AlertRecord, ...]

    def dump(self) -> JsonObject:
        """Return a fresh JSON-compatible snapshot including provenance metadata.

        Returns:
            JsonObject: Detached manifest document.
        """
        return {
            "schema_version": 1, "source": source_metadata(), "taxonomy": taxonomy_metadata(),
            "selection": {
                "seed": self.seed, "algorithm": SELECTION_ALGORITHM,
                "record_order": RECORD_ORDER, "category_order": list(CATEGORY_ORDER),
                "selected_ids": [record.id for record in self.records],
            },
            "base_request_representation": BASE_REPRESENTATION,
            "records": [record.dump() for record in self.records],
        }


def source_record(value: object, name: str) -> AlertRecord:
    data = object_value(value, name, {"id", "prompt", "category"})
    identifier = integer_value(data["id"], f"{name}.id")
    prompt = text_value(data["prompt"], f"{name}.prompt")
    label = text_value(data["category"], f"{name}.category")
    try:
        micro = canonical_category(label)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from None
    return AlertRecord(identifier, prompt, label, micro, MICRO_TO_MACRO[micro], prompt)


def check_coverage(records: Iterable[AlertRecord]) -> None:
    categories = {record.micro_category for record in records}
    if missing := [category for category in CATEGORY_ORDER if category not in categories]:
        raise ValueError(f"Missing ALERT micro categories: {', '.join(missing)}")


def _source_records(rows: Iterable[object]) -> list[AlertRecord]:
    records: list[AlertRecord] = []
    identifiers: set[int] = set()
    for index, row in enumerate(rows, 1):
        record = source_record(row, f"source row {index}")
        if record.id in identifiers:
            raise ValueError(f"Duplicate ALERT source id: {record.id}")
        identifiers.add(record.id)
        records.append(record)
    check_coverage(records)
    return records


def select_alert_records(rows: Iterable[object], *, seed: int = DEFAULT_SEED) -> tuple[AlertRecord, ...]:
    """Validate every source row, then sample one per micro category (no I/O).

    This pure sampler also accepts controlled fixtures. Only preparation verifies
    the official source checksum; sampling alone does not attest provenance.

    Args:
        rows (Iterable[object]): Decoded source rows to validate.
        seed (int, default=DEFAULT_SEED): Deterministic random seed.

    Returns:
        tuple[AlertRecord, ...]: One record per canonical category, in category order.
    """
    _ = integer_value(seed, "selection.seed", maximum=MAX_SEED)
    groups: dict[str, list[AlertRecord]] = {category: [] for category in CATEGORY_ORDER}
    for record in sorted(_source_records(rows), key=lambda record: record.id):
        groups[record.micro_category].append(record)
    rng = random.Random()
    rng.seed(seed, version=2)
    return tuple(rng.choice(groups[category]) for category in CATEGORY_ORDER)
