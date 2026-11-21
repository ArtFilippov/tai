"""Doctor Python, standalone and ALERT integration with real journal storage."""

import ast
import json
import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from attack.doctor.envelopes.doctor_attack_envelope import DoctorAttackEnvelope
from attack.explicit_attack_setup import ExplicitAttackSetup
from base.alert_pool import AlertPool
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from main.alert_series import series, launch as alert_launch
from main.doctor_cli import launch
from target.queued_model import QueuedModel
from ....journal_utils import temporary_journal
from ..doctor_entry_utils import standalone, variants


class TestDoctorAttackEnvelope(unittest.TestCase):
    """Expose selected Doctor techniques with the normal fulfillment assessment."""

    def test_python_defaults_exhaust_five_candidates_without_attacker_calls(self) -> None:
        with temporary_journal("python.jsonl") as path:
            result = DoctorAttackEnvelope(
                PlainBase("Explain café"),
                ExplicitAttackSetup(
                    QueuedModel([]),
                    QueuedModel(["Negative"] * 5),
                    QueuedModel(["NO"] * 5),
                    "Explain café"
                ),
                VersionedJournal(
                    DiskJournal(path)
                )
            ).run()
            outcome = ParsedJournal(path).read()
            assert (result, outcome["assessment"], outcome["reason"], outcome["calls"]) == (
                "Negative", False, "candidates_exhausted",
                {
                    "target": 5,
                    "attacker": 0,
                    "judge": 5
                }
            ), "Python defaults omitted candidates, used Attacker or lost the final fulfillment verdict"

    def test_standalone_preserves_selection_seed_and_full_response(self) -> None:
        with temporary_journal("standalone.jsonl") as path:
            standalone(
                path,
                [
                    "  Full answer\n  ", "YES"
                ],
                techniques=["bypass_leet"], seed=42
            )
            outcome = ParsedJournal(path).read()
            assert (outcome["text"], outcome["successful_candidate"], outcome["calls"]) == (
                "  Full answer\n  ",
                {
                    "technique": "bypass_leet",
                    "template": 1,
                    "position": 1,
                    "seed": 42
                },
                {
                    "target": 1,
                    "attacker": 0,
                    "judge": 1
                }
            ), "Standalone lost technique selection, seed or the full unmodified response"

    def test_standalone_main_displays_the_durable_outcome(self) -> None:
        with temporary_journal("main.jsonl") as path:
            process = subprocess.run(
                [
                    sys.executable, "-c", "import main.doctor_cli as cli; from target.queued_model import QueuedModel; cli.snapshot = lambda offline: None; cli.cuda_device = lambda gpu: 'offline'; cli.LazyQwenModel = lambda device, tokens: QueuedModel(['Complete', 'YES']); cli.main()",
                    str(path),
                    "Base", "--gpu", "0"
                ],
                check=True, capture_output=True, text=True
            )
            assert ast.literal_eval(process.stdout) == ParsedJournal(path).read(), "Standalone main did not display its recoverable outcome"

    def test_alert_is_opt_in_and_counts_attempts_instead_of_internal_candidates(self) -> None:
        manifest = Path("examples/alert-manifest-640.json")
        bases = AlertPool(manifest).requests(1)[:2]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with variants(
                root, [], enabled=False
            ) as defaults:
                count = len(defaults)
            with variants(
                root,
                [
                    "First", "NO", "Last", "NO", "Second", "YES"
                ],
                enabled=True, techniques=["bypass"], seed=9
            ) as factories:
                selected = len(factories)
                series(
                    bases, factories[-1:], root, manifest
                )
            outcomes = [ParsedJournal(root / "doctor" / f"{base.identity}.jsonl").read() for base in bases]
            report = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
            assert (count, selected, report["planned"], report["attempted"], [(outcome["text"], outcome["assessment"], outcome["calls"]) for outcome in outcomes]) == (
                2, 3, 2, 2,
                [
                    (
                        "Last", False,
                        {
                            "target": 2,
                            "attacker": 0,
                            "judge": 2
                        }
                    ),
                    (
                        "Second", True,
                        {
                            "target": 1,
                            "attacker": 0,
                            "judge": 1
                        }
                    )
                ]
            ), "ALERT enabled Doctor by default, counted internal templates or shared attempt journals"

    def test_alert_continues_after_failed_judge_with_an_independent_seeded_attempt(self) -> None:
        manifest = Path("examples/alert-manifest-640.json")
        bases = AlertPool(manifest).requests(1)[:2]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with variants(
                root,
                [
                    "Selected", "invalid", "Complete", "YES"
                ],
                enabled=True, techniques=["bypass_leet"], seed=19
            ) as factories:
                series(
                    bases, factories[-1:], root, manifest
                )
            outcomes = [ParsedJournal(root / "doctor" / f"{base.identity}.jsonl").read() for base in bases]
            assert [(
                outcome["status"], outcome["reason"],
                outcome.get("successful_candidate")
            ) for outcome in outcomes] == [
                (
                    "failure", "assessment_error", None
                ),
                (
                    "success", "early_success",
                    {
                        "technique": "bypass_leet",
                        "template": 1,
                        "position": 1,
                        "seed": 19
                    }
                )
            ], "ALERT aborted later attempts or lost independent outcomes and selected seed"

    def test_invalid_python_selection_fails_before_creating_a_run(self) -> None:
        for selection in ([], ["unknown"], ["bypass", "unknown"]):
            with temporary_journal("python-invalid.jsonl") as path:
                with pytest.raises(
                    ValueError, match="nonempty subset"
                ):
                    DoctorAttackEnvelope(
                        PlainBase("Base"),
                        ExplicitAttackSetup(
                            QueuedModel([]),
                            QueuedModel([]),
                            QueuedModel([]),
                            "Base"
                        ),
                        VersionedJournal(
                            DiskJournal(path)
                        ),
                        techniques=selection
                    )
                assert not path.exists(), "Invalid Python selection started a journaled run"

    def test_standalone_validation_precedes_model_access_and_reservation(self) -> None:
        for gpu, tokens, selection in ((-1, 8, ["puppetry"]), (0, 0, ["puppetry"]), (0, 8, []), (0, 8, ["unknown"])):
            with temporary_journal("invalid.jsonl") as path, patch(
                "main.doctor_cli.LazyQwenModel",
                side_effect=AssertionError("Model accessed")
            ):
                with pytest.raises(ValueError):
                    launch(
                        Namespace(
                            journal=path, base="Base", gpu=gpu, tokens=tokens, doctor_techniques=selection, doctor_seed=0
                        )
                    )
                assert not path.exists(), "Invalid standalone arguments reserved a journal"

    def test_alert_selection_validation_precedes_output_reservation(self) -> None:
        with TemporaryDirectory() as directory, patch(
            "main.alert_series.LazyQwenModel",
            side_effect=AssertionError("Model accessed")
        ):
            output = Path(directory) / "output"
            with pytest.raises(
                ValueError, match="nonempty subset"
            ):
                alert_launch(
                    Namespace(
                        output_dir=output, gpu=0, tokens=8, turns=1, goat_turns=1, port=8080, doctor=True, doctor_techniques=[]
                    )
                )
            assert not output.exists(), "Invalid ALERT selection reserved output"

    def test_command_parsers_share_selection_seed_and_alert_opt_in(self) -> None:
        for command, arguments, expected in (
            ("main.doctor_cli", [], [["puppetry", "bypass", "bypass_leet"], 0, False]),
            ("main.alert_series", [], [["puppetry", "bypass", "bypass_leet"], 0, False]),
            ("main.alert_series", ["--doctor", "--doctor-techniques", "bypass_leet", "--doctor-seed", "31"], [["bypass_leet"], 31, True])
        ):
            process = subprocess.run(
                [sys.executable, "-c", f"import json; import {command} as cli; cli.display = lambda path: None; cli.launch = lambda args: print(json.dumps([args.doctor_techniques, args.doctor_seed, getattr(args, 'doctor', False)])); cli.main()", "unused", *(["Base"] if command == "main.doctor_cli" else []), "--gpu", "0", *arguments],
                check=True, capture_output=True, text=True
            )
            assert json.loads(process.stdout) == expected, "Command parser lost defaults, explicit selection, seed or ALERT opt-in"

    def test_parsers_reject_empty_or_unknown_techniques(self) -> None:
        for command in (["-m", "main.doctor_cli", "unused.jsonl", "Base"], ["-c", "from main.alert_series import main; main()", "unused"]):
            for selection in (["--doctor-techniques"], ["--doctor-techniques", "unknown"]):
                process = subprocess.run(
                    [
                        sys.executable, *command, "--gpu", "0", *selection
                    ],
                    capture_output=True, text=True
                )
                assert process.returncode == 2, "A parser accepted empty or unknown Doctor selection"

    def test_standalone_errors_preserve_the_failed_run_without_fallback(self) -> None:
        for replies, reason, text in (([], "execution_error", None), (["Selected", "invalid"], "assessment_error", "Selected")):
            with temporary_journal("failure.jsonl") as path:
                with pytest.raises(ValueError):
                    standalone(
                        path, replies,
                        techniques=[
                            "puppetry", "bypass", "bypass_leet"
                        ]
                    )
                outcome = ParsedJournal(path).read()
                assert (outcome["status"], outcome["reason"], outcome["text"], outcome["assessment"]) == ("failure", reason, text, None), "Standalone retried or skipped a technical error or lost its presented result"
