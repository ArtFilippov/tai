"""Explicit network/local preparation, separate from the offline manifest loader."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from .manifest import check_taxonomy
from .records import (
    DEFAULT_SEED,
    SOURCE_COUNT,
    SOURCE_SHA256,
    SOURCE_SIZE,
    SOURCE_URL,
    AlertManifest,
    MAX_SEED,
    select_alert_records,
)
from .values import decode_json_lines, integer_value


def _validate_paths(source: Path, output: Path, taxonomy: Path) -> None:
    paths = (source, output, taxonomy)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Source, output and taxonomy must be different files")
    check_taxonomy(taxonomy)
    if not output.parent.is_dir() or not source.parent.is_dir():
        raise ValueError("Source and output parent directories must exist")


def _source_bytes(source_file: Path, download: bool) -> bytes:
    if download:
        with urlopen(SOURCE_URL, timeout=60) as response:
            raw: bytes = response.read()
    else:
        raw = source_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError(f"Source SHA-256 mismatch; expected {SOURCE_SHA256}")
    if len(raw) != SOURCE_SIZE:
        raise ValueError(f"Source size mismatch; expected {SOURCE_SIZE} bytes")
    return raw


def prepare_alert(
    source_file: Path,
    output: Path,
    *,
    taxonomy_path: Path,
    seed: int = DEFAULT_SEED,
    download: bool = False,
) -> AlertManifest:
    """Verify pinned source bytes and all rows before writing the sample.

    With download=True, also save the verified official JSONL to source_file.
    Otherwise source_file is read locally, with no network fallback.

    Args:
        source_file (Path): Pinned source JSONL, or destination for a download.
        output (Path): Destination for the sampled manifest.
        taxonomy_path (Path): Local pinned taxonomy document.
        seed (int, default=DEFAULT_SEED): Deterministic sampling seed.
        download (bool, default=False): Whether to explicitly download the official source.

    Returns:
        AlertManifest: Validated selected records.

    Raises:
        ValueError: The source row count differs from the pinned source.
    """
    _validate_paths(source_file, output, taxonomy_path)
    _ = integer_value(seed, "selection.seed", maximum=MAX_SEED)
    raw = _source_bytes(source_file, download)
    rows = decode_json_lines(raw.decode("utf-8"))
    records = select_alert_records(rows, seed=seed)
    if len(rows) != SOURCE_COUNT:
        raise ValueError(f"Source record count mismatch; expected {SOURCE_COUNT}")
    manifest = AlertManifest(seed, records)
    content = (json.dumps(manifest.dump(), ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    if download:
        _ = source_file.write_bytes(raw)
    _ = output.write_bytes(content)
    return manifest
