"""Deterministic 20-per-category selection and complete offline validation."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from alert import load_alert_manifest, select_alert_records
from alert import preparation
from alert.records import DEFAULT_SEED, LEGACY_SELECTED_IDS
from alert.taxonomy import CATEGORY_ORDER

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "ALERT_taxonomy.md"
MANIFEST = ROOT / "examples/alert-manifest-20.json"
LEGACY = ROOT / "examples/alert-manifest.json"
MANIFEST_SHA256 = "eb8a2bfb79a43345958502ff4b0b59b603a0a541f6318c1766dc81def23186fa"


@pytest.fixture
def rows() -> list[dict[str, object]]:
    return [
        {"id": category_index * 100 + variant, "prompt": f"text {category_index}/{variant}\r\n", "category": category}
        for category_index, category in enumerate(CATEGORY_ORDER)
        for variant in range(21)
    ]


def test_extended_sampler_keeps_original_choice_and_is_deterministic(rows: list[dict[str, object]]) -> None:
    state = random.getstate()
    legacy = select_alert_records(rows)
    selected = select_alert_records(rows, per_category=20)
    assert random.getstate() == state
    assert selected == select_alert_records(reversed(rows), per_category=20)
    assert selected != select_alert_records(rows, seed=DEFAULT_SEED + 1, per_category=20)
    assert len(selected) == len({record.id for record in selected}) == 640
    assert [record.micro_category for record in selected] == [category for category in CATEGORY_ORDER for _ in range(20)]
    assert all(legacy[index] in selected[index * 20:(index + 1) * 20] for index in range(32))
    source = {row["id"]: row for row in rows}
    assert all(record.base_request == record.prompt == source[record.id]["prompt"] for record in selected)


def test_extended_sampler_validates_full_source_and_capacity(rows: list[dict[str, object]]) -> None:
    rows[0]["prompt"] = " "
    with pytest.raises(ValueError, match="source row 1.prompt"):
        select_alert_records(rows, per_category=20)
    rows[0]["prompt"] = "restored"
    with pytest.raises(ValueError, match="Insufficient distinct ALERT records for hate_body"):
        select_alert_records(rows[1:19] + rows[21:], per_category=20)
    with pytest.raises(ValueError, match="Missing ALERT micro categories: hate_body"):
        select_alert_records(rows[21:], per_category=20)
    with pytest.raises(ValueError, match="Duplicate ALERT source id"):
        select_alert_records([*rows, rows[0]], per_category=20)
    with pytest.raises(ValueError, match="selection.per_category"):
        select_alert_records(rows, per_category=19)
    with pytest.raises(ValueError, match="selection.per_category"):
        select_alert_records(rows, per_category=20.0)  # type: ignore[arg-type]


def test_official_extended_manifest_is_pinned_and_legacy_still_loads() -> None:
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == MANIFEST_SHA256
    original = load_alert_manifest(LEGACY, taxonomy_path=TAXONOMY)
    manifest = load_alert_manifest(MANIFEST, taxonomy_path=TAXONOMY)
    assert manifest.per_category == 20 and original.per_category == 1
    assert tuple(record.id for record in original.records) == LEGACY_SELECTED_IDS
    assert len(manifest.records) == len({record.id for record in manifest.records}) == 640
    assert Counter(record.micro_category for record in manifest.records) == dict.fromkeys(CATEGORY_ORDER, 20)
    for index, legacy in enumerate(original.records):
        assert legacy in manifest.records[index * 20:(index + 1) * 20]


@pytest.mark.parametrize("change", ["duplicate", "count", "order", "text", "both_texts", "metadata", "legacy", "ids"])
def test_extended_loader_rejects_invalid_document(tmp_path: Path, change: str) -> None:
    document = json.loads(MANIFEST.read_bytes())
    records = document["records"]
    selection = document["selection"]
    if change == "duplicate":
        records[1]["id"] = records[0]["id"]
    elif change == "count":
        records.pop()
    elif change == "order":
        records[1], records[20] = records[20], records[1]
        selection["selected_ids"] = [record["id"] for record in records]
    elif change == "text":
        records[-1]["base_request"] += "altered"
    elif change == "both_texts":
        records[-1]["prompt"] += "altered"
        records[-1]["base_request"] += "altered"
    elif change == "metadata":
        document["source"]["revision"] = "main"
    elif change == "legacy":
        records[0]["id"] = 0
        selection["selected_ids"][0] = 0
    else:
        selection["selected_ids"][0] = 0
    path = tmp_path / "altered.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_alert_manifest(path, taxonomy_path=TAXONOMY)


def test_extended_preparation_is_byte_stable_and_does_not_publish_on_failure(
    rows: list[dict[str, object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    monkeypatch.setattr(preparation, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(preparation, "SOURCE_SIZE", len(raw))
    monkeypatch.setattr(preparation, "SOURCE_COUNT", len(rows))
    monkeypatch.setattr(preparation, "urlopen", lambda *args, **kwargs: pytest.fail("unexpected network access"))
    source = tmp_path / "source.jsonl"
    output = tmp_path / "manifest.json"
    source.write_bytes(raw)
    preparation.prepare_alert(source, output, taxonomy_path=TAXONOMY, sampling=(DEFAULT_SEED, 20))
    first = output.read_bytes()
    preparation.prepare_alert(source, output, taxonomy_path=TAXONOMY, sampling=(DEFAULT_SEED, 20))
    assert output.read_bytes() == first
    source.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preparation.prepare_alert(source, output, taxonomy_path=TAXONOMY, sampling=(DEFAULT_SEED, 20))
    assert output.read_bytes() == first
