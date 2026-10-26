"""A root summary keeps only known outcomes across attack factories."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from attack.attack import Attack
from attack.envelopes.fitd_attack_envelope import FITDAttackEnvelope
from attack.fitd_setup import FITDSetup
from base.alert_pool import AlertPool
from base.base_request import BaseRequest
from benchmark.alert_series import AlertSeries
from journal.parsed_journal import ParsedJournal
from main.alert_series import journal, series
from ..attack.model_fixture_utils import refusal


class TestAlertSeries(unittest.TestCase):
    """Check order, independent failures, and a recoverable root summary."""

    def example(self, base: BaseRequest, label: str) -> Attack:
        """Supply one controlled result for a factory and a request.

        Args:
            base (BaseRequest): Request selected by the loop.
            label (str): Factory behavior.

        Returns:
            Attack: Controlled result and description.
        """
        attack = Mock(spec=Attack)
        attack.description.return_value = {"type": label}
        attack.run.return_value = base.text()
        if base.metadata()["microcategory"] == "hate_disabled" and label in {"failure", "interrupt"}:
            attack.run.side_effect = ValueError("unavailable") if label == "failure" else KeyboardInterrupt()
        return attack

    def report(self, directory: str) -> dict[str, Any]:
        """Read the saved root summary.

        Args:
            directory (str): Temporary launch directory.

        Returns:
            dict[str, Any]: Recorded attempts.
        """
        path = Path(directory) / "summary.json"
        return json.loads(
            path.read_text(encoding="utf-8")
        )

    def test_factory_first_order(self) -> None:
        bases = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)[:2]
        with tempfile.TemporaryDirectory() as directory:
            series(
                bases,
                [
                    lambda base: self.example(base, "first"), lambda base: self.example(base, "second")
                ],
                Path(directory),
                Path("manifest.json")
            )
            runs = self.report(directory)["runs"]
            assert [(run["factory"], run["base"]) for run in runs] == [
                (
                    1,
                    bases[0].text()
                ),
                (
                    1,
                    bases[1].text()
                ),
                (
                    2,
                    bases[0].text()
                ),
                (
                    2,
                    bases[1].text()
                )
            ], "The series did not exhaust each factory before advancing to the next"

    def test_records_only_available_fields(self) -> None:
        bases = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)[:1]
        with tempfile.TemporaryDirectory() as directory:
            series(
                bases, [lambda base: self.example(base, "first")],
                Path(directory),
                Path("manifest.json")
            )
            run = self.report(directory)["runs"][0]
            assert run == {
                "number": 1,
                "factory": 1,
                "base": bases[0].text(),
                "base_metadata": dict(
                    bases[0].metadata()
                ),
                "attack": {"type": "first"},
                "result": bases[0].text(),
                "outcome": "completed"
            }, "The root index invented an assessment or omitted the presented result"

    def test_error_does_not_stop_other_factories(self) -> None:
        bases = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)[:2]
        with tempfile.TemporaryDirectory() as directory:
            series(
                bases,
                [
                    lambda base: self.example(base, "failure"), lambda base: self.example(base, "next")
                ],
                Path(directory),
                Path("manifest.json")
            )
            report = self.report(directory)
            assert (report["attempted"], report["complete"], report["runs"][-1]["attack"]) == (
                4, True, {"type": "next"}
            ), "A failed attempt stopped the remaining factories"

    def test_factory_failure_has_no_invented_attack(self) -> None:
        bases = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)[:1]

        def failure(base: BaseRequest) -> Attack:
            raise ValueError("startup unavailable")

        with tempfile.TemporaryDirectory() as directory:
            series(
                bases,
                [
                    failure, lambda base: self.example(base, "next")
                ],
                Path(directory),
                Path("manifest.json")
            )
            report = self.report(directory)
            assert (report["runs"][0]["outcome"], report["runs"][0]["error"], "attack" in report["runs"][0], report["runs"][1]["outcome"]) == (
                "failure", "ValueError('startup unavailable')", False, "completed"
            ), "A factory failure invented an attack or stopped the next factory"

    def test_real_fitd_preserves_source_requests(self) -> None:
        bases = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)[:2]
        model = refusal()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            series(
                bases, [lambda base: FITDAttackEnvelope( base, FITDSetup(model, model, model, base.text()), journal(root / "fitd" / f"{base.metadata()['id']}.jsonl"), 1 )], root,
                Path("manifest.json")
            )
            path = root / "fitd" / f"{bases[1].identity}.jsonl"
            report = self.report(directory)
            assert (
                list(ParsedJournal(path).events())[0]["base_metadata"],
                report["runs"][1]["attack"]["type"], report["attempted"]
            ) == (
                dict(
                    bases[1].metadata()
                ),
                "fitd", 2
            ), "The shared models mixed requests across FITD attempts"

    def test_interruption_keeps_partial_summary(self) -> None:
        bases = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)[:2]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt):
                series(
                    bases, [lambda base: self.example(base, "interrupt")],
                    Path(directory),
                    Path("manifest.json")
                )
            report = self.report(directory)
            assert (report["complete"], report["attempted"], report["runs"][-1]["outcome"]) == (
                False, 2, "interrupted"
            ), "An interrupted run was counted as complete"

    def test_empty_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = AlertSeries(
                Path("manifest.json"),
                Path(directory),
                0
            )
            assert summary.report()["complete"] is True, "An empty plan was marked incomplete"
