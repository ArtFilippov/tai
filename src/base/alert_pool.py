"""Offline validation and selection of pinned ALERT base requests."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, override

from base.alert_identity import AlertIdentity
from base.alert_taxonomy_contract import AlertTaxonomyContract

MANIFEST_SHA256 = "3ee64c7a6b86c2b28174d8c699407e24f94a4e54ca9b8f9c5fc35bf460725f43"
SOURCE = {
    "rows": 14763,
    "repo": "Babelscape/ALERT",
    "config": "alert",
    "file": "alert.jsonl",
    "split": "test",
    "revision": "c76d0637f4573742982d5fe961a31bdb3d1e2c80",
    "sha256": "0a5594d47d3c4b81462048e6e5dfa22304be6eb490019ddbddc73adc4e317cc4",
}
TAXONOMY = {
    "document": "ALERT_taxonomy.md",
    "version": "ALERT Section 3 Figure 2",
    "sha256": "0877011641dbe3a558a9b6e917d65c1daf103b1722eece04bd56905b4ba0debd",
}
SELECTION = {
    "method": "lowest-source-id",
    "per_category": 20,
    "text": "original prompt"
}
GROUPS_KEY = "groups"
COUNT = 20
TOTAL = 640


class AlertPool(AlertTaxonomyContract):
    """A complete, validated pool of original prompts per Figure 2 category."""

    class Request(AlertIdentity):
        """One selected original prompt and its pinned source provenance."""

        def __init__(self, record: Mapping[str, Any], source: Mapping[str, Any], taxonomy: Mapping[str, Any]) -> None:
            """Hold one original source record and its provenance.

            Args:
                record (Mapping[str, Any]): Selected source request.
                source (Mapping[str, Any]): Pinned source provenance.
                taxonomy (Mapping[str, Any]): Pinned taxonomy provenance.
            """
            super().__init__()
            self.record: Mapping[str, str | int] = record
            self.source: Mapping[str, str | int] = source
            self.taxonomy: Mapping[str, str | int] = taxonomy

        @override
        def text(self) -> str:
            return str(self.record["prompt"])

        @property
        @override
        def identity(self) -> int:
            """Return the original source ID."""
            return int(self.record["id"])

        @property
        @override
        def microcategory(self) -> str:
            """Return the canonical Figure 2 microcategory."""
            return str(self.record["microcategory"])

        @property
        @override
        def macro(self) -> str:
            """Return the canonical Figure 2 macrocategory."""
            return str(self.record["macro"])

        @override
        def metadata(self) -> Mapping[str, str | int]:
            return {
                "id": self.record["id"],
                "source_category": self.record["source_category"],
                "microcategory": self.record["microcategory"],
                "macro": self.record["macro"],
                "source_revision": self.source["revision"],
                "source_sha256": self.source["sha256"],
                "taxonomy_version": self.taxonomy["version"],
                "taxonomy_sha256": self.taxonomy["sha256"],
            }

    def __init__(self, manifest: Path) -> None:
        """Remember the local pinned manifest until requests are selected.

        Args:
            manifest (Path): Local copy of the pinned manifest.
        """
        super().__init__()
        self.manifest = manifest

    def taxonomy(self) -> dict[str, dict[str, str | list[str]]]:
        return self.groups()

    def requests(self, count: object) -> list[AlertIdentity]:
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= COUNT:
            raise ValueError("ALERT count must be between 1 and 20")
        content = self.manifest.read_bytes()
        manifest: dict[str, Any] = json.loads(content)
        bases = self.verify(
            manifest, content
        )
        return [
            self.Request(record, manifest["source"], manifest["taxonomy"])
            for category in self.categories() for record in bases[category][:count]
        ]

    def verify(self, manifest: dict[str, Any], content: bytes) -> dict[str, list[dict[str, Any]]]:
        self.provenance(manifest)
        bases: dict[str, list[dict[str, Any]]] = {category: [] for category in self.categories()}
        identities: set[int] = set()
        for record in manifest["requests"]:
            self.record(
                record, bases, identities
            )
        for category, items in bases.items():
            if len(items) != COUNT:
                raise ValueError(f"ALERT category {category!r} has {len(items)} requests, expected 20")
        if len(identities) != TOTAL:
            raise ValueError("ALERT manifest must contain exactly 640 unique source IDs")
        if [record["id"] for record in manifest["requests"]] != [
            record["id"] for category in self.categories() for record in bases[category]
        ]:
            raise ValueError("ALERT manifest is not in canonical category order")
        if hashlib.sha256(content).hexdigest() != MANIFEST_SHA256:
            raise ValueError("ALERT manifest checksum mismatch")
        return bases

    def provenance(self, manifest: dict[str, Any]) -> None:
        """Check the declared source, taxonomy and selection before individual records.

        Args:
            manifest (dict[str, Any]): Candidate offline manifest.

        Raises:
            ValueError: If its pinned provenance or taxonomy is wrong.
        """
        if manifest["source"] != SOURCE:
            raise ValueError("ALERT source provenance does not match the pinned revision")
        if {key: value for key, value in manifest["taxonomy"].items() if key != GROUPS_KEY} != TAXONOMY:
            raise ValueError("ALERT taxonomy checksum does not match ALERT_taxonomy.md")
        self.validate(manifest["taxonomy"]["groups"])
        if manifest["selection"] != SELECTION:
            raise ValueError("Unknown ALERT selection method")

    def record(
        self, record: dict[str, Any], bases: dict[str, list[dict[str, Any]]], identities: set[int],
    ) -> None:
        if set(record) != {"id", "prompt", "sha256", "source_category", "microcategory", "macro"}:
            raise ValueError("Malformed ALERT request fields")
        identity = record["id"]
        prompt = record["prompt"]
        if isinstance(identity, bool) or not isinstance(identity, int) or identity < 0 or identity in identities:
            raise ValueError(f"Invalid or repeated ALERT source ID: {identity!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing ALERT prompt for ID {identity}")
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != record["sha256"]:
            raise ValueError(f"ALERT prompt checksum mismatch for ID {identity}")
        category = self.canonical(record["source_category"])
        if record["microcategory"] != category or record["macro"] != self.macro(category):
            raise ValueError(f"ALERT category mapping mismatch for ID {identity}")
        if bases[category] and identity <= bases[category][-1]["id"]:
            raise ValueError(f"ALERT IDs are not ordered in category {category!r}")
        identities.add(identity)
        bases[category].append(record)
