"""Explicit offline preparation of a pinned ALERT JSONL into a local manifest.

Download separately with huggingface_hub.hf_hub_download(repo_id="Babelscape/ALERT",
filename="alert.jsonl", repo_type="dataset",
revision="c76d0637f4573742982d5fe961a31bdb3d1e2c80"). Then run:
python -m main.alert_manifest_cli /path/to/alert.jsonl examples/alert-manifest-640.json
The file checksum and all 14,763 records are verified before writing anything.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from base.alert_taxonomy_contract import AlertTaxonomyContract

SOURCE_SHA256 = "0a5594d47d3c4b81462048e6e5dfa22304be6eb490019ddbddc73adc4e317cc4"
SOURCE_ROWS = 14763
PER_CATEGORY = 20

class AlertManifest(AlertTaxonomyContract):
    """Builds the same ordered pool from an independently downloaded source file."""

    def __init__(self, source: Path) -> None:
        """Keep the pinned source path for explicit offline preparation.

        Args:
            source (Path): Local official JSONL snapshot.
        """
        super().__init__()
        self.source = source

    def build(self) -> dict[str, Any]:
        content = self.source.read_bytes()
        if (digest := hashlib.sha256(content).hexdigest()) != SOURCE_SHA256:
            raise ValueError("ALERT source checksum does not match pinned alert.jsonl")
        rows = [json.loads(line) for line in content.splitlines()]
        if len(rows) != SOURCE_ROWS:
            raise ValueError(f"ALERT source has {len(rows)} rows, expected {SOURCE_ROWS}")
        selected: dict[str, list[dict[str, Any]]] = {category: [] for category in self.categories()}
        identities: set[int] = set()
        for row in rows:
            self.record(
                row, selected, identities
            )
        for category, items in selected.items():
            if len(items) < PER_CATEGORY:
                raise ValueError(f"ALERT category {category!r} has only {len(items)} source requests")
        return {
            "source": {
                "repo": "Babelscape/ALERT",
                "config": "alert",
                "split": "test",
                "revision": "c76d0637f4573742982d5fe961a31bdb3d1e2c80",
                "file": "alert.jsonl",
                "sha256": digest,
                "rows": len(rows)
            },
            "taxonomy": {
                "document": "ALERT_taxonomy.md",
                "version": "ALERT Section 3 Figure 2",
                "sha256": "0877011641dbe3a558a9b6e917d65c1daf103b1722eece04bd56905b4ba0debd",
                "groups": self.groups()
            },
            "selection": {
                "method": "lowest-source-id",
                "per_category": PER_CATEGORY,
                "text": "original prompt"
            },
            "requests": [item for category in self.categories() for item in sorted(selected[category], key=lambda item: item["id"])[:PER_CATEGORY]]
        }

    def record(
        self, row: dict[str, Any], selected: dict[str, list[dict[str, Any]]],
        identities: set[int],
    ) -> None:
        if set(row) != {"id", "prompt", "category"}:
            raise ValueError("Malformed ALERT source record")
        identity = row["id"]
        prompt = row["prompt"]
        if isinstance(identity, bool) or not isinstance(identity, int) or identity < 0 or identity in identities:
            raise ValueError(f"Invalid or repeated ALERT source ID: {identity!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing ALERT prompt for ID {identity}")
        category = self.canonical(row["category"])
        identities.add(identity)
        selected[category].append(
            {
                "id": identity,
                "prompt": prompt,
                "sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "source_category": row["category"],
                "microcategory": category,
                "macro": self.macro(category)
            }
        )

    def save(self, destination: Path) -> None:
        _ = destination.write_text(
            json.dumps(
                self.build(),
                ensure_ascii=False, indent=2
            ) + "\n", encoding="utf-8"
        )
