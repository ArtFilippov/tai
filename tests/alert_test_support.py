"""Typed documents used to deliberately mutate ALERT validation fixtures."""

from typing import TypedDict


class ManifestDocument(TypedDict):
    schema_version: object
    source: dict[str, object]
    taxonomy: dict[str, object]
    selection: dict[str, object]
    records: list[dict[str, object]]
    base_request_representation: object


class SavedIndex(TypedDict):
    schema_version: object
    series_id: str
    manifest: str
    planned: list[dict[str, object]]
    entries: list[dict[str, object]]
    finished_at: str
    interrupted: bool
    complete: bool
