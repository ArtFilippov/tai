"""Offline sampling, pre-inference validation, and the shipped official sample."""

import hashlib
import io
import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from tai import AlertManifest, load_alert_manifest, select_alert_records
from tai.alert import preparation
from tai.alert.manifest import DEFAULT_SEED, SOURCE_COUNT, SOURCE_SHA256, SOURCE_URL
from tai.alert.taxonomy import CATEGORY_ORDER, MACRO_CATEGORIES, TAXONOMY_SHA256
from tai.app.cli import main

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "ALERT_taxonomy.md"
MANIFEST = ROOT / "examples/alert-manifest.json"
MANIFEST_SHA256 = "fb398308890b68f2ad1565cb46d9277189289315775a7c4bc05c3c8fb3a2b42b"
GROUP_COUNTS = {
    "hate": 8,
    "crime": 8,
    "substance": 5,
    "sex": 3,
    "self_harm": 3,
    "weapon": 5,
}


@pytest.fixture
def rows():
    return [
        {
            "id": index * 10 + variant,
            "prompt": f"### Instruction: test {index}/{variant} π\r\n### Response: ",
            "category": category,
        }
        for index, category in enumerate(CATEGORY_ORDER)
        for variant in (1, 2, 10)
    ]


def test_sampling_has_golden_numeric_order_and_does_not_touch_global_rng(rows):
    state = random.getstate()
    selected = select_alert_records(rows)
    assert random.getstate() == state
    assert selected == select_alert_records(reversed(rows))
    assert selected != select_alert_records(rows, seed=DEFAULT_SEED + 1)
    assert [record.id for record in selected] == [
        10,
        20,
        30,
        31,
        50,
        51,
        61,
        72,
        90,
        91,
        102,
        112,
        121,
        140,
        142,
        160,
        161,
        171,
        181,
        191,
        201,
        211,
        222,
        240,
        241,
        252,
        270,
        272,
        282,
        300,
        310,
        311,
    ]
    source = {row["id"]: row for row in rows}
    assert tuple(record.micro_category for record in selected) == CATEGORY_ORDER
    assert Counter(record.macro_category for record in selected) == GROUP_COUNTS
    for record in selected:
        assert record.base_request == record.prompt == source[record.id]["prompt"]
        assert record.source_category == source[record.id]["category"]


@pytest.mark.parametrize(
    ("label", "canonical", "macro"),
    [
        ("crime_kidnap", "crime_kidnapp", "crime"),
        ("crime-kidnapping", "crime_kidnapp", "crime"),
        ("crime-kidnap", "crime_kidnapp", "crime"),
        ("crime-kidnapp", "crime_kidnapp", "crime"),
        ("self-harm-pro-thin", "self_harm_thin", "self_harm"),
        ("self-harm-thin", "self_harm_thin", "self_harm"),
        ("hate-lgbtq+", "hate_lgbtq+", "hate"),
    ],
)
def test_known_aliases_preserve_original_labels_in_sampler_and_loader(
    rows, tmp_path, label, canonical, macro
):
    for row in rows:
        if row["category"] == canonical:
            row["category"] = label
    selected = select_alert_records(rows)
    record = next(record for record in selected if record.micro_category == canonical)
    assert record.source_category == label
    assert record.macro_category == macro
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(AlertManifest(DEFAULT_SEED, selected).dump()))
    assert load_alert_manifest(path, taxonomy_path=TAXONOMY).records == selected


def test_all_prose_hyphen_labels_are_explicitly_supported(rows):
    for row in rows:
        row["category"] = row["category"].replace("_", "-")
    selected = select_alert_records(rows)
    assert tuple(record.micro_category for record in selected) == CATEGORY_ORDER
    assert all(record.source_category != record.micro_category for record in selected)


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("category", "hate_invented", "Unknown ALERT category: 'hate_invented'"),
        ("category", "HATE_BODY", "Unknown ALERT category"),
        ("category", " hate_body", "Unknown ALERT category"),
        ("category", None, "category: expected nonempty text"),
        ("prompt", " \t\n", "prompt: expected nonempty text"),
        ("prompt", 123, "prompt: expected nonempty text"),
        ("prompt", "\ud800", "prompt: expected valid Unicode"),
        ("id", True, "id: expected a nonnegative integer"),
        ("id", -1, "id: expected a nonnegative integer"),
        ("id", "123", "id: expected a nonnegative integer"),
        ("id", 1.5, "id: expected a nonnegative integer"),
    ],
)
def test_bad_source_rows_fail_even_when_they_would_not_be_selected(
    rows, field, value, diagnostic
):
    rows[0][field] = value  # golden selection chooses row 3 in this category
    with pytest.raises(ValueError, match=re.escape(diagnostic)):
        select_alert_records(rows)


def test_duplicate_source_ids_and_missing_categories_are_diagnosed(rows):
    with pytest.raises(ValueError, match="Duplicate ALERT source id: 1"):
        select_alert_records([*rows, rows[0]])
    missing = {"crime_kidnapp", "self_harm_thin"}
    incomplete = [row for row in rows if row["category"] not in missing]
    with pytest.raises(ValueError, match="crime_kidnapp, self_harm_thin"):
        select_alert_records(incomplete)


@pytest.mark.parametrize(
    "row",
    [None, [], {}, {"id": 1, "prompt": "text", "category": "hate_body", "extra": 1}],
)
def test_source_schema_errors(row, rows):
    with pytest.raises(ValueError, match="source row 97"):
        select_alert_records([*rows, row])


@pytest.mark.parametrize("seed", [True, -1, 1.5, "42", 2**64])
def test_invalid_seed(seed, rows):
    with pytest.raises(ValueError, match="selection.seed"):
        select_alert_records(rows, seed=seed)


@pytest.fixture
def document():
    return json.loads(MANIFEST.read_bytes())


def load_modified(document, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return load_alert_manifest(path, taxonomy_path=TAXONOMY)


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("prompt", "", "prompt: expected nonempty text"),
        ("prompt", ["text"], "prompt: expected nonempty text"),
        ("base_request", "changed", "base_request"),
        ("source_category", "weapon_unknown", "Unknown ALERT category"),
        ("source_category", None, "source_category|category"),
        ("micro_category", "hate_other", "micro_category"),
        ("macro_category", "crime", "macro_category"),
        ("id", True, "id: expected a nonnegative integer"),
        ("id", SOURCE_COUNT, "id: expected a nonnegative integer"),
    ],
)
def test_loader_rejects_bad_records_before_returning_any_input(
    document, tmp_path, field, value, diagnostic
):
    document["records"][-1][field] = value
    with pytest.raises(ValueError, match=diagnostic):
        load_modified(document, tmp_path)


def test_loader_names_missing_category(document, tmp_path):
    document["records"].pop(10)
    with pytest.raises(
        ValueError, match="Missing ALERT micro categories: crime_kidnapp"
    ):
        load_modified(document, tmp_path)


def test_loader_rejects_duplicate_ids(document, tmp_path):
    document["records"][1]["id"] = document["records"][0]["id"]
    with pytest.raises(ValueError, match="Duplicate selected ALERT id: 5861"):
        load_modified(document, tmp_path)


def test_loader_rejects_duplicate_categories(document, tmp_path):
    duplicate = dict(document["records"][0], id=0)
    document["records"].append(duplicate)
    with pytest.raises(
        ValueError, match="Duplicate selected ALERT micro category: hate_body"
    ):
        load_modified(document, tmp_path)


def test_loader_rejects_reordered_records_even_if_id_list_is_updated(
    document, tmp_path
):
    document["records"].reverse()
    document["selection"]["selected_ids"].reverse()
    with pytest.raises(ValueError, match="records order"):
        load_modified(document, tmp_path)


@pytest.mark.parametrize(
    ("path", "value", "diagnostic"),
    [
        (("schema_version",), True, "schema_version"),
        (("schema_version",), 2, "schema_version"),
        (
            ("base_request_representation",),
            "extract-instruction",
            "base representation",
        ),
        (("source", "revision"), "main", "source metadata"),
        (("source", "config"), "alert_adversarial", "source metadata"),
        (("source", "split"), "train", "source metadata"),
        (("source", "sha256"), "0" * 64, "source metadata"),
        (("source", "record_count"), 14763.0, "source metadata"),
        (("taxonomy", "version"), "new", "taxonomy metadata"),
        (("taxonomy", "sha256"), "0" * 64, "taxonomy metadata"),
        (("taxonomy", "aliases"), {}, "taxonomy metadata"),
        (("taxonomy", "macro_categories"), {}, "taxonomy metadata"),
        (("selection", "seed"), True, "selection.seed"),
        (("selection", "algorithm"), "random", "selection.algorithm"),
        (("selection", "record_order"), "lexicographic", "selection.record_order"),
        (("selection", "category_order"), [], "selection.category_order"),
        (("selection", "selected_ids"), [], "selection.selected_ids"),
        (("records",), {}, "records: expected a JSON array"),
        (("records",), [], "Missing ALERT micro categories"),
    ],
)
def test_loader_checks_provenance_and_sampling_contract(
    document, tmp_path, path, value, diagnostic
):
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=diagnostic):
        load_modified(document, tmp_path)


@pytest.mark.parametrize("section", ["root", "record", "selection"])
@pytest.mark.parametrize("operation", ["missing", "unknown"])
def test_loader_rejects_schema_drift(document, tmp_path, section, operation):
    target = {
        "root": document,
        "record": document["records"][0],
        "selection": document["selection"],
    }[section]
    if operation == "missing":
        target.pop(next(iter(target)))
    else:
        target["unexpected"] = "field"
    with pytest.raises(ValueError, match="missing fields.*unknown fields"):
        load_modified(document, tmp_path)


@pytest.mark.parametrize(
    "content",
    [b"not JSON", b"\xff", b'{"schema_version":1,"schema_version":1}', b'{"seed":NaN}'],
)
def test_invalid_json_is_rejected(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        load_alert_manifest(path, taxonomy_path=TAXONOMY)


def test_loader_verifies_actual_taxonomy_bytes_and_missing_files(tmp_path):
    taxonomy = tmp_path / "taxonomy.md"
    taxonomy.write_bytes(TAXONOMY.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="taxonomy SHA-256 mismatch"):
        load_alert_manifest(MANIFEST, taxonomy_path=taxonomy)
    with pytest.raises(FileNotFoundError):
        load_alert_manifest(tmp_path / "absent.json", taxonomy_path=TAXONOMY)
    with pytest.raises(FileNotFoundError):
        load_alert_manifest(MANIFEST, taxonomy_path=tmp_path / "absent.md")


def test_shipped_official_manifest_and_taxonomy_are_pinned_and_complete():
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == MANIFEST_SHA256
    assert hashlib.sha256(TAXONOMY.read_bytes()).hexdigest() == TAXONOMY_SHA256
    # Compare the executable mapping/order to the supplied Figure 2 descriptions.
    identifiers = re.findall(r"^- `([^`]+)` —", TAXONOMY.read_text(), re.MULTILINE)
    assert tuple(identifiers) == CATEGORY_ORDER
    assert {
        group: len(micros) for group, micros in MACRO_CATEGORIES.items()
    } == GROUP_COUNTS
    manifest = load_alert_manifest(MANIFEST, taxonomy_path=TAXONOMY)
    assert manifest.seed == 20260915
    assert (
        len(manifest.records) == len({record.id for record in manifest.records}) == 32
    )
    assert Counter(record.macro_category for record in manifest.records) == GROUP_COUNTS
    assert tuple(record.micro_category for record in manifest.records) == tuple(
        identifiers
    )
    assert all(record.prompt == record.base_request for record in manifest.records)
    assert all(
        record.prompt.startswith("### Instruction:") for record in manifest.records
    )
    assert manifest.records[10].source_category == "crime_kidnap"
    assert manifest.records[10].micro_category == "crime_kidnapp"
    snapshot = manifest.dump()
    assert snapshot["source"]["sha256"] == SOURCE_SHA256
    assert snapshot == json.loads(MANIFEST.read_bytes())
    snapshot["records"].clear()
    assert len(manifest.records) == len(manifest.dump()["records"]) == 32


@pytest.fixture
def controlled_source(rows, monkeypatch):
    """Pin synthetic bytes inside preparation tests; no network or external data."""
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    monkeypatch.setattr(preparation, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(preparation, "SOURCE_SIZE", len(raw))
    monkeypatch.setattr(preparation, "SOURCE_COUNT", len(rows))
    return raw


def test_preparation_download_then_offline_reproduces_identical_bytes(
    controlled_source, tmp_path, monkeypatch, capsys
):
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(controlled_source)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Offline preparation must not download or construct a model profile"
        )

    monkeypatch.setattr(preparation, "urlopen", download)
    monkeypatch.setattr("tai.app.cli.load_profile", forbidden)
    source = tmp_path / "alert.jsonl"
    output = tmp_path / "sample.json"
    args = [
        "prepare-alert",
        "--source-file",
        str(source),
        "--output",
        str(output),
        "--taxonomy",
        str(TAXONOMY),
        "--seed",
        str(DEFAULT_SEED),
    ]
    assert main([*args, "--download"]) == 0
    assert calls == [(SOURCE_URL, 60)]
    assert source.read_bytes() == controlled_source
    original = output.read_bytes()
    monkeypatch.setattr(preparation, "urlopen", forbidden)
    assert main(args) == 0
    assert output.read_bytes() == original
    assert len(load_alert_manifest(output, taxonomy_path=TAXONOMY).records) == 32
    assert "32 записей, seed=20260915" in capsys.readouterr().out


def test_preparation_rejects_wrong_source_without_writing_output(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.json"
    source.write_bytes(b"wrong source")
    output.write_bytes(b"previous manifest")
    with pytest.raises(ValueError, match="Source SHA-256 mismatch"):
        preparation.prepare_alert(source, output, taxonomy_path=TAXONOMY)
    monkeypatch.setattr(
        preparation, "urlopen", lambda *a, **kw: io.BytesIO(b"bad download")
    )
    with pytest.raises(ValueError, match="Source SHA-256 mismatch"):
        preparation.prepare_alert(source, output, taxonomy_path=TAXONOMY, download=True)
    assert source.read_bytes() == b"wrong source"
    assert output.read_bytes() == b"previous manifest"


def test_preparation_local_errors_never_fall_back_to_network(
    tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        pytest.fail("No implicit network fallback")

    monkeypatch.setattr(preparation, "urlopen", forbidden)
    source = tmp_path / "missing.jsonl"
    output = tmp_path / "output.json"
    assert (
        main(
            [
                "prepare-alert",
                "--source-file",
                str(source),
                "--output",
                str(output),
                "--taxonomy",
                str(TAXONOMY),
            ]
        )
        == 2
    )
    assert "missing.jsonl" in capsys.readouterr().err
    assert not output.exists()
    with pytest.raises(ValueError, match="different files"):
        preparation.prepare_alert(
            TAXONOMY, output, taxonomy_path=TAXONOMY, download=True
        )
