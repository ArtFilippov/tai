"""The pinned pool must fail closed before any model is used."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest

from base.alert_manifest import AlertManifest
from base.alert_pool import AlertPool
from base.alert_taxonomy import AlertTaxonomy
from base.base_request import BaseRequest


class TestAlertPool(unittest.TestCase):
    """The published manifest and corrupted offline copies exercise the boundary."""

    def manifest(self) -> dict[str, Any]:
        return json.loads(
            Path("examples/alert-manifest-640.json").read_text(encoding="utf-8")
        )

    def save(self, data: dict[str, Any], directory: str) -> Path:
        path = Path(directory) / "manifest.json"
        path.write_text(
            json.dumps(data),
            encoding="utf-8"
        )
        return path

    def test_coverage(self) -> None:
        pool = AlertPool(
            Path("examples/alert-manifest-640.json")
        )
        assert len(pool.requests(20)) == 640, "The complete pool was not available offline"

    def test_first_of_each_category(self) -> None:
        pool = AlertPool(
            Path("examples/alert-manifest-640.json")
        )
        assert [base.metadata()["microcategory"] for base in pool.requests(1)] == AlertTaxonomy().categories(), "Categories were not served in Figure 2 order"

    def test_first_two_from_each_category(self) -> None:
        pool = AlertPool(
            Path("examples/alert-manifest-640.json")
        )
        assert [base.metadata()["id"] for base in pool.requests(2)] == [row["id"] for index, row in enumerate(self.manifest()["requests"]) if index % 20 < 2], "The first two source IDs were not selected per category"

    def test_full_taxonomy(self) -> None:
        pool = AlertPool(
            Path("examples/alert-manifest-640.json")
        )
        assert pool.taxonomy() == self.manifest()["taxonomy"]["groups"], "The pool did not expose the complete pinned taxonomy"

    def test_original_text(self) -> None:
        pool = AlertPool(
            Path("examples/alert-manifest-640.json")
        )
        assert pool.requests(1)[0].text() == self.manifest()["requests"][0]["prompt"], "The source prompt was changed"

    def test_contract(self) -> None:
        base: BaseRequest = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)[0]
        assert base.metadata()["id"] == 5697, "An ALERT request did not expose its source ID"

    def test_pinned_provenance(self) -> None:
        base = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)[0]
        assert base.metadata() == {
            "id": 5697,
            "source_category": "hate_body",
            "microcategory": "hate_body",
            "macro": "hate",
            "source_revision": "c76d0637f4573742982d5fe961a31bdb3d1e2c80",
            "source_sha256": "0a5594d47d3c4b81462048e6e5dfa22304be6eb490019ddbddc73adc4e317cc4",
            "taxonomy_version": "ALERT Section 3 Figure 2",
            "taxonomy_sha256": "0877011641dbe3a558a9b6e917d65c1daf103b1722eece04bd56905b4ba0debd",
        }, "The request did not retain the pinned source and taxonomy provenance"

    def test_source_alias(self) -> None:
        pool = AlertPool(
            Path("examples/alert-manifest-640.json")
        )
        assert pool.requests(20)[8 * 20 + 2 * 20].metadata()["source_category"] == "crime_kidnap", "The original kidnapping category was lost"

    def test_aliases(self) -> None:
        assert [AlertTaxonomy().canonical(label) for label in ("crime_kidnap", "crime-kidnapping", "self-harm-pro-thin")] == ["crime_kidnapp", "crime_kidnapp", "self_harm_thin"], "An explicit paper alias was not recognized"

    def test_unknown_alias(self) -> None:
        with pytest.raises(
            ValueError, match="Unknown ALERT category"
        ):
            AlertTaxonomy().canonical("crime_mystery")

    def test_nontext_category(self) -> None:
        data = self.manifest()
        data["requests"][0]["source_category"] = ["hate_body"]
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(
                ValueError, match="Unknown ALERT category"
            ):
                AlertPool(
                    self.save(
                        data, directory
                    )
                ).requests(1)

    def test_missing_category(self) -> None:
        data = self.manifest()
        data["requests"] = [row for row in data["requests"] if row["microcategory"] != "weapon_other"]
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(
                ValueError, match="weapon_other.*0 requests"
            ):
                AlertPool(
                    self.save(
                        data, directory
                    )
                ).requests(1)

    def test_unknown_source_category(self) -> None:
        data = self.manifest()
        data["requests"][0]["source_category"] = "hate_unknown"
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(
                ValueError, match="Unknown ALERT category"
            ):
                AlertPool(
                    self.save(
                        data, directory
                    )
                ).requests(1)

    def test_duplicate_id(self) -> None:
        data = self.manifest()
        data["requests"][1]["id"] = data["requests"][0]["id"]
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(
                ValueError, match="repeated ALERT source ID"
            ):
                AlertPool(
                    self.save(
                        data, directory
                    )
                ).requests(1)

    def test_changed_prompt(self) -> None:
        data = self.manifest()
        data["requests"][0]["prompt"] = "different"
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(
                ValueError, match="prompt checksum mismatch"
            ):
                AlertPool(
                    self.save(
                        data, directory
                    )
                ).requests(1)

    def test_changed_prompt_with_matching_internal_checksum(self) -> None:
        data = self.manifest()
        data["requests"][0]["prompt"] = "different"
        data["requests"][0]["sha256"] = (
            hashlib.sha256(b"different")
            .hexdigest()
        )
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(
                ValueError, match="manifest checksum mismatch"
            ):
                AlertPool(
                    self.save(
                        data, directory
                    )
                ).requests(1)

    def test_changed_taxonomy(self) -> None:
        data = self.manifest()
        data["taxonomy"]["groups"]["crime"]["microcategories"][2] = "crime_kidnap"
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(
                ValueError, match="does not match Figure 2"
            ):
                AlertPool(
                    self.save(
                        data, directory
                    )
                ).requests(1)

    def test_bad_order(self) -> None:
        data = self.manifest()
        data["requests"][:40] = data["requests"][20:40] + data["requests"][:20]
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(
                ValueError, match="not in canonical category order"
            ):
                AlertPool(
                    self.save(
                        data, directory
                    )
                ).requests(1)

    def test_bad_count(self) -> None:
        with pytest.raises(
            ValueError, match="between 1 and 20"
        ):
            AlertPool(
                Path("examples/alert-manifest-640.json")
            ).requests(21)

    def test_unpinned_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "alert.jsonl"
            source.write_text(
                '{"id": 0}\n', encoding="utf-8"
            )
            with pytest.raises(
                ValueError, match="source checksum"
            ):
                AlertManifest(source).build()

    def test_manifest_digest(self) -> None:
        digest = (
            hashlib.sha256(
                Path("examples/alert-manifest-640.json").read_bytes()
            ).hexdigest()
        )
        assert digest == "3ee64c7a6b86c2b28174d8c699407e24f94a4e54ca9b8f9c5fc35bf460725f43", "The committed pool changed without review"
