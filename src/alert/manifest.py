"""Offline ALERT manifest loading; no downloads or model calls."""

import hashlib
from pathlib import Path

from .records import (
    BASE_REPRESENTATION, DEFAULT_SEED, RECORD_ORDER, SELECTION_ALGORITHM,
    SOURCE_COUNT, SOURCE_REVISION, SOURCE_SHA256, SOURCE_SIZE, SOURCE_URL,
    AlertManifest, AlertRecord, select_alert_records,
)
from .validation import validate_manifest
from .taxonomy import TAXONOMY_SHA256
from .values import decode_json

__all__ = [
    "BASE_REPRESENTATION", "DEFAULT_SEED", "RECORD_ORDER", "SELECTION_ALGORITHM",
    "SOURCE_COUNT", "SOURCE_REVISION", "SOURCE_SHA256", "SOURCE_SIZE",
    "SOURCE_URL", "AlertManifest", "AlertRecord", "select_alert_records", "load_alert_manifest",
    "check_taxonomy",
]


def check_taxonomy(path: Path) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != TAXONOMY_SHA256:
        raise ValueError(f"{path}: taxonomy SHA-256 mismatch; expected {TAXONOMY_SHA256}")


def load_alert_manifest(path: Path, *, taxonomy_path: Path) -> AlertManifest:
    """Load and validate the complete v1 manifest and local taxonomy before inference.

    No source download, model construction, or lazy per-row validation occurs.
    ValueError diagnoses invalid content; OSError diagnoses missing/unreadable files.
    Source membership is verified during preparation, not by this offline loader.

    Args:
        path (Path): Saved JSON manifest.
        taxonomy_path (Path): Pinned local taxonomy document.

    Returns:
        AlertManifest: Validated records in experiment input order.
    """
    check_taxonomy(taxonomy_path)
    return validate_manifest(decode_json(path.read_bytes().decode("utf-8")))
