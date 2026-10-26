"""Offline checks of the ALERT selection, saved outcomes and isolated FITD roles."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

from attack.model_run import ModelRun
from attack.run_capture import RunCapture
from base.alert_pool import AlertPool
from base.alert_taxonomy import AlertTaxonomy
from base.base_request import BaseRequest
from benchmark.alert_archive import AlertArchive
from benchmark.alert_series import AlertSeries
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from main.alert_series import archive, launch, outcome
from main.alert_series import run as qwen_run
from target.model import Model
from ..attack.model_fixture_utils import refusal
from .alert_fixture_utils import assessment, attempt, interrupted, scripted


MANIFEST = Path("examples/alert-manifest-640.json")


class TestAlertSeries(unittest.TestCase):
    """Exercise the full pinned pool without network, CUDA or model weights."""

    def test_limit_one(self) -> None:
        pool = AlertPool(MANIFEST)
        assert len(pool.requests(1)) == 32, "The first-per-category limit did not select 32 requests"

    def test_limit_twenty(self) -> None:
        pool = AlertPool(MANIFEST)
        assert len(pool.requests(20)) == 640, "The default-sized pool did not select 640 requests"

    def test_short_offline_series_counts(self) -> None:
        pool = AlertPool(MANIFEST)
        outcomes: dict[Path, dict[str, str]] = {}
        memory = Mock(spec=AlertArchive)
        memory.categories.return_value = AlertTaxonomy().categories()
        memory.capture.side_effect = RunCapture
        memory.outcome.side_effect = outcomes.__getitem__

        def record(base: BaseRequest, path: Path, model: Model, attacker: Model, *, turns: int) -> None:
            outcomes[path] = {
                "outcome": "successful",
                "base": str(
                    base.metadata()["id"]
                )
            }

        with tempfile.TemporaryDirectory() as directory:
            series = AlertSeries(
                pool.requests(1),
                pool.taxonomy(),
                Path(directory),
                MANIFEST, archive=memory
            )
            series.run(
                attempt(record)
            )
            summary = series.report()
            assert (
                summary["complete"], summary["attempted"],
                tuple(group["planned"] for group in summary["macro"].values()),
                len({entry["journal"] for entry in summary["runs"]}),
                memory.save.call_count
            ) == (True, 32, (8, 8, 5, 3, 3, 5), 32, 32), "The short series lost an attempt, category, or summary"

    def test_full_pool_plan(self) -> None:
        pool = AlertPool(MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            summary = AlertSeries(
                pool.requests(20),
                pool.taxonomy(),
                Path(directory),
                MANIFEST, archive=archive()
            ).report()
            assert (
                summary["planned"],
                [group["planned"] for group in summary["macro"].values()]
            ) == (640, [160, 160, 100, 60, 60, 100]), "The complete pool lost its macro-category plan"

    def test_invalid_limit_precedes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unused"
            with self.assertRaisesRegex(
                ValueError, "between 1 and 20"
            ):
                launch(
                    Namespace(
                        output_dir=path, manifest=MANIFEST, gpu=0, limit=21, turns=1, tokens=1, port=8080
                    )
                )
            assert not path.exists(), "An invalid limit created output before preflight"

    def test_existing_output_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            marker = path / "summary.json"
            _ = marker.write_text(
                "previous", encoding="utf-8"
            )
            with self.assertRaises(FileExistsError):
                launch(
                    Namespace(
                        output_dir=path, manifest=MANIFEST, gpu=0, limit=1, turns=1, tokens=1, port=8080
                    )
                )
            assert marker.read_text(encoding="utf-8") == "previous", "An existing summary was overwritten"

    def test_canonical_order(self) -> None:
        pool = AlertPool(MANIFEST)
        assert [base.metadata()["microcategory"] for base in pool.requests(1)] == [
            category for group in pool.taxonomy().values() for category in group["microcategories"]
        ], "The requests lost their canonical category order"

    def test_report_covers_full_taxonomy(self) -> None:
        pool = AlertPool(MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            summary = AlertSeries(
                pool.requests(1),
                pool.taxonomy(),
                Path(directory),
                MANIFEST, archive=archive()
            ).report()
            assert (
                list(summary["micro"]),
                list(summary["macro"])
            ) == (
                [category for group in pool.taxonomy().values() for category in group["microcategories"]],
                ["hate", "crime", "substance", "sex", "self_harm", "weapon"],
            ), "The summary omitted or reordered a canonical category"

    def test_aggregate_and_continuation(self) -> None:
        pool = AlertPool(MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            series = AlertSeries(
                pool.requests(1)[:4],
                pool.taxonomy(),
                Path(directory),
                MANIFEST, archive=archive()
            )
            series.run(
                attempt(scripted)
            )
            path = Path(directory) / "summary.json"
            text = path.read_text(encoding="utf-8")
            summary = json.loads(text)
            assert (
                summary["macro"]["hate"]["planned"], summary["macro"]["hate"]["evaluated"], summary["macro"]["hate"]["successful"], summary["macro"]["hate"]["unsuccessful"], summary["macro"]["hate"]["failure"], summary["macro"]["hate"]["no_result"], summary["macro"]["hate"]["rate"],
                summary["attempted"]
            ) == (4, 2, 1, 1, 1, 1, 50.0, 4), "The saved outcomes or error continuation were miscounted"

    def test_empty_macro(self) -> None:
        pool = AlertPool(MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            series = AlertSeries(
                pool.requests(1)[:4],
                pool.taxonomy(),
                Path(directory),
                MANIFEST, archive=archive()
            )
            series.run(
                attempt(scripted)
            )
            assert series.count("weapon", key="macro")["rate"] is None, "An unassessed group acquired a rate"

    def test_startup_error(self) -> None:
        pool = AlertPool(MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            series = AlertSeries(
                pool.requests(1)[:3],
                pool.taxonomy(),
                Path(directory),
                MANIFEST, archive=archive()
            )
            series.run(
                attempt(scripted)
            )
            entry = series.report()["runs"][1]
            path = Path(directory) / str(entry["journal"])
            assert list(ParsedJournal(path).events())[1]["event"] == "startup_error", "The error before start was not saved"

    def test_startup_error_keeps_series_reference(self) -> None:
        pool = AlertPool(MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            series = AlertSeries(
                pool.requests(1)[:2],
                pool.taxonomy(),
                Path(directory),
                MANIFEST,
                archive=archive()
            )
            series.run(
                attempt(scripted)
            )
            path = (Path(directory) /
                    str(
                        series.report()["runs"][1]["journal"]
                    ))
            assert list(ParsedJournal(path).events())[0]["summary"] == "summary.json", "An early failure lost its series reference"

    def test_interruption_keeps_series_reference(self) -> None:
        pool = AlertPool(MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            series = AlertSeries(
                pool.requests(1)[:2],
                pool.taxonomy(),
                Path(directory),
                MANIFEST,
                archive=archive()
            )
            with self.assertRaises(KeyboardInterrupt):
                series.run(
                    attempt(interrupted)
                )
            path = (Path(directory) /
                    str(
                        series.report()["runs"][1]["journal"]
                    ))
            assert list(ParsedJournal(path).events())[0]["summary"] == "summary.json", "An interrupted run lost its series reference"

    def test_invalid_final_judgment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            journal = VersionedJournal(
                DiskJournal(path)
            )
            journal.record(
                {
                    "event": "start",
                    "base": "request"
                }
            )
            journal.record(
                {
                    "event": "result",
                    "text": "first"
                }
            )
            journal.record(
                {
                    "event": "judgment",
                    "text": "first",
                    "assessment": True
                }
            )
            journal.record(
                {
                    "event": "result",
                    "text": "last"
                }
            )
            journal.record(
                {
                    "event": "judgment",
                    "text": "last",
                    "assessment": "YES"
                }
            )
            journal.record(
                {
                    "event": "end",
                    "status": "success"
                }
            )
            assert outcome(path)["outcome"] == "invalid", "A stale verdict made an invalid final judgment successful"

    def test_judge_error_is_distinct_from_execution_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judge.jsonl"
            journal = VersionedJournal(
                DiskJournal(path)
            )
            journal.record(
                {
                    "event": "start",
                    "base": "request"
                }
            )
            journal.record(
                {
                    "event": "result",
                    "text": "first"
                }
            )
            journal.record(
                {
                    "event": "judgment",
                    "text": "first",
                    "assessment": False
                }
            )
            journal.record(
                {
                    "event": "result",
                    "text": "last"
                }
            )
            journal.record(
                {
                    "event": "assessment_error",
                    "text": "last",
                    "error": "invalid verdict"
                }
            )
            journal.record(
                {
                    "event": "end",
                    "status": "failure",
                    "reason": "assessment_error"
                }
            )
            assert outcome(path)["outcome"] == "invalid", "A judge error was treated as a negative or successful verdict"

    def test_interruption_keeps_partial_summary(self) -> None:
        pool = AlertPool(MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            series = AlertSeries(
                pool.requests(1)[:3],
                pool.taxonomy(),
                Path(directory),
                MANIFEST, archive=archive()
            )
            with self.assertRaises(KeyboardInterrupt):
                series.run(
                    attempt(interrupted)
                )
            path = Path(directory) / "summary.json"
            text = path.read_text(encoding="utf-8")
            summary = json.loads(text)
            assert (summary["complete"], summary["attempted"], summary["macro"]["hate"]["unevaluated"]) == (False, 2, 2), "An interrupted run was counted as completed"

    def test_last_interruption_is_incomplete(self) -> None:
        pool = AlertPool(MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            series = AlertSeries(
                pool.requests(1)[:2],
                pool.taxonomy(),
                Path(directory),
                MANIFEST, archive=archive()
            )
            with self.assertRaises(KeyboardInterrupt):
                series.run(
                    attempt(interrupted)
                )
            assert not series.report()["complete"], "An interrupted final attempt completed the series"

    def test_thrown_assessment_error_is_invalid(self) -> None:
        pool = AlertPool(MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            series = AlertSeries(
                pool.requests(1)[:1],
                pool.taxonomy(),
                Path(directory),
                MANIFEST, archive=archive()
            )
            series.run(
                attempt(assessment)
            )
            assert series.report()["micro"]["hate_body"]["invalid"] == 1, "A thrown invalid evaluation was counted as execution failure"

    def test_archive_metadata_with_real_fitd(self) -> None:
        pool = AlertPool(MANIFEST)
        model = refusal()
        with tempfile.TemporaryDirectory() as directory:
            series = AlertSeries(
                pool.requests(1)[:2],
                pool.taxonomy(),
                Path(directory),
                MANIFEST, archive=archive()
            )
            series.run(
                ModelRun(
                    model, model, 1,
                    qwen_run
                )
            )
            paths = [Path(directory) / str(entry["journal"]) for entry in series.report()["runs"]]
            first = list(
                ParsedJournal(paths[0]).events()
            )[0]
            second = list(
                ParsedJournal(paths[1]).events()
            )[0]
            assert (
                first["summary"] == second["summary"] == "summary.json"
                and list(ParsedJournal(paths[0]).events())[1]["base_metadata"] == dict(pool.requests(1)[0].metadata())
                and list(ParsedJournal(paths[1]).events())[1]["base_metadata"] == dict(pool.requests(1)[1].metadata())
                and list(ParsedJournal(paths[0]).events())[1]["attack"] == list(ParsedJournal(paths[1]).events())[1]["attack"]
                and any(event["event"] == "call" for event in ParsedJournal(paths[0]).events())
            ), "Separate FITD journals lost their archived base or attack settings"
