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
EXTENDED_COUNT = 20
EXTENDED_ALGORITHM = "python3.12-random-mt19937-choice-plus-sample-v1"
# The original official manifest's selected_ids, in Figure 2 order.
LEGACY_SELECTED_IDS = (
    5861, 5936, 6195, 7579, 8847, 8974, 9706, 7631,
    2582, 827, 4657, 2754, 4927, 4417, 3882, 1879,
    10222, 11776, 10675, 12031, 11405, 12613, 12871, 12260,
    13144, 13308, 13469, 14514, 13713, 13625, 14679, 14329,
)


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
        per_category (int, default=1): Records selected per micro category (1 or 20).
    """

    seed: int
    records: tuple[AlertRecord, ...]
    per_category: int = 1

    def dump(self) -> JsonObject:
        """Return a fresh JSON-compatible snapshot including provenance metadata.

        Returns:
            JsonObject: Detached manifest document.
        """
        extended = self.per_category == EXTENDED_COUNT
        selection: JsonObject = {
                "seed": self.seed, "algorithm": EXTENDED_ALGORITHM if extended else SELECTION_ALGORITHM,
                "record_order": RECORD_ORDER, "category_order": list(CATEGORY_ORDER),
                "selected_ids": [record.id for record in self.records],
        }
        if extended:
            selection["per_category"] = EXTENDED_COUNT
            selection["legacy_selected_ids"] = list(LEGACY_SELECTED_IDS)
        return {
            "schema_version": 2 if extended else 1, "source": source_metadata(), "taxonomy": taxonomy_metadata(),
            "selection": selection,
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


def _extended_selection(groups: dict[str, list[AlertRecord]], seed: int) -> tuple[AlertRecord, ...]:
    rng = random.Random(seed)
    legacy_rng = random.Random(DEFAULT_SEED)
    selected: list[AlertRecord] = []
    for category in CATEGORY_ORDER:
        group = groups[category]
        if len(group) < EXTENDED_COUNT:
            raise ValueError(f"Insufficient distinct ALERT records for {category}: {len(group)} < 20")
        legacy = legacy_rng.choice(group)
        remaining = [record for record in group if record.id != legacy.id]
        selected.extend((legacy, *rng.sample(remaining, EXTENDED_COUNT - 1)))
    return tuple(selected)


def select_alert_records(
    rows: Iterable[object], *, seed: int = DEFAULT_SEED, per_category: int = 1,
) -> tuple[AlertRecord, ...]:
    """Validate every source row, then sample one or 20 per category (no I/O).

    This pure sampler also accepts controlled fixtures. Only preparation verifies
    the official source checksum; sampling alone does not attest provenance.

    Args:
        rows (Iterable[object]): Decoded source rows to validate.
        seed (int, default=DEFAULT_SEED): Deterministic random seed.
        per_category (int, default=1): Sample size, either 1 or 20.

    Returns:
        tuple[AlertRecord, ...]: Records grouped in canonical category order.

    Raises:
        ValueError: Invalid source row, seed, sample size or insufficient records.
    """
    _ = integer_value(seed, "selection.seed", maximum=MAX_SEED)
    _ = integer_value(per_category, "selection.per_category", maximum=EXTENDED_COUNT)
    if per_category not in (1, EXTENDED_COUNT):
        raise ValueError("selection.per_category: expected 1 or 20")
    groups: dict[str, list[AlertRecord]] = {category: [] for category in CATEGORY_ORDER}
    for record in sorted(_source_records(rows), key=lambda record: record.id):
        groups[record.micro_category].append(record)
    if per_category == 1:
        rng = random.Random(seed)
        return tuple(rng.choice(groups[category]) for category in CATEGORY_ORDER)
    return _extended_selection(groups, seed)
