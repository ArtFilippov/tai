"""A root summary keeps only known outcomes across attack factories."""

import json
from argparse import Namespace
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from attack.attack import Attack
from attack.envelopes.fitd_attack_envelope import FITDAttackEnvelope
from attack.envelopes.adaptive_attacks_attack_envelope import AdaptiveAttacksAttackEnvelope
from attack.envelopes.goat_attack_envelope import GOATAttackEnvelope
from attack.envelopes.threaten_json_attack_envelope import ThreatenJSONAttackEnvelope
from attack.fitd_setup import FITDSetup
from base.alert_pool import AlertPool
from base.base_request import BaseRequest
from base.plain_base import PlainBase
from benchmark.alert_series import AlertSeries
from journal.parsed_journal import ParsedJournal
from main.alert_series import journal, prepare, series
from ..attack.model_fixture_utils import adaptive, goat, refusal, threaten_json
from ..target import test_logits, test_model


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

    def test_offline_goat_has_independent_alert_journals(self) -> None:
        bases = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)[:2]
        model = goat()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            series(
                bases, [lambda base: GOATAttackEnvelope( base, FITDSetup(model, model, model, base.text()), journal(root / "goat" / f"{base.metadata()['id']}.jsonl"), 1 )], root,
                Path("manifest.json")
            )
            events = list(
                ParsedJournal(root / "goat" / f"{bases[1].identity}.jsonl").events()
            )
            assert (
                self.report(directory)["attempted"],
                events[0]["base"], [event["role"] for event in events if event["event"] == "call"]
            ) == (
                        2,
                        bases[1].text(),
                        [
                            "attacker", "target", "attacker", "judge"
                        ]
                    ), "The offline GOAT series lost an independent run or mixed roles"

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

    def test_offline_threaten_json_has_independent_alert_journals(self) -> None:
        bases = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)[:2]
        model = threaten_json()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            series(
                bases, [lambda base: ThreatenJSONAttackEnvelope( base, FITDSetup(model, model, model, base.text()), journal(root / "threaten_json" / f"{base.metadata()['id']}.jsonl") )], root,
                Path("manifest.json")
            )
            path = root / "threaten_json" / f"{bases[1].identity}.jsonl"
            events = list(
                ParsedJournal(path).events()
            )
            assert (
                self.report(directory)["attempted"],
                events[0]["base"], [event["phase"] for event in events if event["event"] == "call" and event["role"] == "target"]
            ) == (
                2,
                bases[1].text(),
                [
                    "control", "pressured"
                ]
            ), "The ALERT variant mixed independent requests or conversations"

    def test_alert_flag_wires_offline_threaten_json_factory(self) -> None:
        base = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)[0]
        arguments = Namespace(
            gpu=0, tokens=192, attacker="qwen", host="127.0.0.1", port=8080,
            turns=1, goat_turns=1, no_threaten_json=False, autodan=False, autodan_hga=False
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "main.alert_series.LazyQwenModel", return_value=threaten_json()
        ), patch(
            "main.alert_series.cuda_device", return_value="controlled"
        ), patch(
            "main.alert_series.snapshot", return_value=None
        ), prepare(
            arguments,
            Path(directory)
        ) as factories:
            _ = factories[2](base).run()
            assert ParsedJournal(Path(directory) / "threaten_json" / f"{base.metadata()['id']}.jsonl").read()["text"] == '"Detailed café explanation"', "The ALERT CLI factory did not run its ThreatenJSON variant"

    def test_offline_autodan_series_keeps_separate_journals(self) -> None:
        bases = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)[:2]
        arguments = Namespace(
            gpu=0, tokens=192, attacker="qwen", host="127.0.0.1", port=8080,
            turns=1, goat_turns=1, no_threaten_json=True, autodan=True, autodan_hga=True,
            autodan_target="Desired completion", autodan_steps=1,
            autodan_batch=3, autodan_elites=1, autodan_points=1,
            autodan_crossover=0.0, autodan_mutation=0.0, autodan_seed=8
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "main.alert_series.LazyQwenModel", return_value=test_model.TestModel()
        ), patch(
            "main.alert_series.QwenLogits", return_value=test_logits.TestLogits()
        ), patch(
            "main.alert_series.cuda_device", return_value="controlled"
        ), patch(
            "main.alert_series.snapshot", return_value=None
        ), prepare(
            arguments,
            Path(directory)
        ) as factories:
            series(
                bases, factories[-2:],
                Path(directory),
                Path("manifest.json")
            )
            journal_path = Path(directory) / "autodan" / f"{bases[1].identity}.jsonl"
            hga_path = Path(directory) / "autodan_hga" / f"{bases[1].identity}.jsonl"
            assert (
                self.report(directory)["attempted"],
                list(ParsedJournal(journal_path).events())[0]["base"],
                ParsedJournal(journal_path).read()["assessment"],
                list(ParsedJournal(hga_path).events())[0]["base"],
                "'type': 'autodan_hga'" in str(list(ParsedJournal(hga_path).events())[0]["attack"])
            ) == (
                4,
                bases[1].text(),
                True,
                bases[1].text(),
                True
            ), "The offline GA/HGA series mixed requests or attempt journals"

    def test_empty_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = AlertSeries(
                Path("manifest.json"),
                Path(directory),
                0
            )
            assert summary.report()["complete"] is True, "An empty plan was marked incomplete"

    def test_adaptive_series_covers_every_category_offline(self) -> None:
        bases = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)
        arguments = Namespace(
            gpu=0, tokens=192, attacker="qwen", host="127.0.0.1", port=8080,
            turns=1, goat_turns=1, no_threaten_json=False,
            no_adaptive_attacks=False, autodan=False, autodan_hga=False
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "main.alert_series.LazyQwenModel", return_value=adaptive()
        ), patch(
            "main.alert_series.cuda_device", return_value="controlled"
        ), patch(
            "main.alert_series.snapshot", return_value=None
        ), prepare(
            arguments,
            Path(directory)
        ) as factories:
            series(
                bases, factories[-1:],
                Path(directory),
                Path("examples/alert-manifest-640.json")
            )
            report = self.report(directory)
            assert (
                report["attempted"], report["complete"],
                len({run["base_metadata"]["microcategory"] for run in report["runs"]}),
                all(( ParsedJournal(Path(directory) / "adaptive_attacks" / f"{base.identity}.jsonl").read()["assessment"] is False for base in bases ))
            ) == (32, True, 32, True), "The offline adaptive series lost a category, journal, or assessment"

    def test_adaptive_envelope_repeated_runs_keep_separate_bases(self) -> None:
        model = adaptive()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for base in ("First request", "Second request"):
                _ = AdaptiveAttacksAttackEnvelope(
                    PlainBase(base),
                    FITDSetup(
                        model, model, model, base
                    ),
                    journal(root / f"{base}.jsonl")
                ).run()
            assert [list(ParsedJournal(root / f"{base}.jsonl").events())[0]["base"] for base in ("First request", "Second request")] == [
                "First request", "Second request"
            ], "Repeated adaptive runs mixed bases or journals"
