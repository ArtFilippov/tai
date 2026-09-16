"""Explicit network/local preparation, separate from the offline manifest loader."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from tai.alert import (
    DEFAULT_SEED,
    SOURCE_COUNT,
    SOURCE_SHA256,
    SOURCE_SIZE,
    SOURCE_URL,
    AlertManifest,
    _check_taxonomy,
    _decode_json,
    _integer,
    select_alert_records,
)


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
    """
    if len({path.resolve() for path in (source_file, output, taxonomy_path)}) != 3:
        raise ValueError("Source, output and taxonomy must be different files")
    _check_taxonomy(taxonomy_path)
    _integer(seed, "selection.seed", maximum=2**64 - 1)
    if not output.parent.is_dir() or not source_file.parent.is_dir():
        raise ValueError("Source and output parent directories must exist")
    if download:
        with urlopen(SOURCE_URL, timeout=60) as response:
            raw = response.read()
    else:
        raw = source_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError(f"Source SHA-256 mismatch; expected {SOURCE_SHA256}")
    if len(raw) != SOURCE_SIZE:
        raise ValueError(f"Source size mismatch; expected {SOURCE_SIZE} bytes")
    rows = []
    for index, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        try:
            rows.append(_decode_json(line))
        except ValueError as exc:
            raise ValueError(f"Source line {index}: {exc}") from None
    records = select_alert_records(rows, seed=seed)
    if len(rows) != SOURCE_COUNT:
        raise ValueError(f"Source record count mismatch; expected {SOURCE_COUNT}")
    manifest = AlertManifest(seed, records)
    content = (json.dumps(manifest.dump(), ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    if download:
        source_file.write_bytes(raw)
    output.write_bytes(content)
    return manifest
