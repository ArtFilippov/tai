"""Independent past/future results through the public Python assembly."""

import unittest
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import torch

from attack.explicit_attack_setup import ExplicitAttackSetup
from attack.phrasing.envelopes.disk_phrasing_pair_envelope import DiskPhrasingPairEnvelope
from base.plain_base import PlainBase
from base.alert_pool import AlertPool
from journal.parsed_journal import ParsedJournal
from main.alert_series import launch as alert_launch, series
from main.phrasing_cli import launch
from target.queued_model import QueuedModel
from target.lazy_model import LazyModel
from target.local_qwen_assets import LocalQwenAssets
from target.model import Model
from ..phrasing_entry_utils import resources, variants


class TestPhrasingAttackEnvelope(unittest.TestCase):
    """Run both tenses even when the first succeeds or fails technically."""

    def test_python_pair_keeps_independent_successes_for_the_same_reformulation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "pair"
            outcomes = DiskPhrasingPairEnvelope(
                PlainBase("Original"),
                ExplicitAttackSetup(
                    QueuedModel(["Question", "Question"]),
                    QueuedModel(["Past", "Future"]),
                    QueuedModel(["YES", "YES"]),
                    "Original"
                ),
                root, attempts=1
            ).run()
            assert [(
                outcomes[name]["text"], outcomes[name]["assessment"],
                ParsedJournal(root / f"{name}.jsonl").read()["successful_candidate"]
            ) for name in ("phrasing_past", "phrasing_future")] == [
                (
                    "Past", True,
                    {
                        "tense": "past",
                        "generation": 1,
                        "position": 1
                    }
                ),
                (
                    "Future", True,
                    {
                        "tense": "future",
                        "generation": 1,
                        "position": 1
                    }
                )
            ], "The pair skipped future after success, shared deduplication or lost separate outcomes"

    def test_default_budget_and_sampling_policy_are_preserved_in_both_journals(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "pair"
            outcomes = DiskPhrasingPairEnvelope(
                PlainBase("Original"),
                ExplicitAttackSetup(
                    QueuedModel(["Question"] * 40),
                    QueuedModel(["Past", "Future"]),
                    QueuedModel(["NO", "NO"]),
                    "Original"
                ),
                root
            ).run()
            starts = [json.loads((root / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()[0])["attack"]["probe"] for name in ("phrasing_past", "phrasing_future")]
            assert [(outcomes[name]["calls"], starts[index]["attempts"], starts[index]["attacker"]["model"]["generation"]) for index, name in enumerate(("phrasing_past", "phrasing_future"))] == [
                (
                    {
                        "attacker": 20,
                        "target": 1,
                        "judge": 1
                    },
                    20,
                    {
                        "max_new_tokens": 150,
                        "do_sample": True,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": 0,
                        "seed": 0,
                        "randomness": "advancing run-local stream"
                    }
                )
            ] * 2, "A tense lost its bounded default budget, independent sampling settings or duplicate skips"

    def test_python_pair_runs_future_after_empty_past_preparation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "pair"
            outcomes = DiskPhrasingPairEnvelope(
                PlainBase("Original"),
                ExplicitAttackSetup(
                    QueuedModel([' "" ', "Question"]),
                    QueuedModel(["Future"]),
                    QueuedModel(["YES"]),
                    "Original"
                ),
                root, attempts=2
            ).run()
            assert [(outcomes[name]["status"], outcomes[name]["reason"], outcomes[name]["text"]) for name in ("phrasing_past", "phrasing_future")] == [
                ("failure", "preparation_error", None), ("success", "early_success", "Future")
            ], "Failed preparation skipped future or converted past into a negative verdict"

    def test_prestart_model_loading_failures_are_recoverable_in_a_fresh_process(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "pair"
            models: list[Model] = []
            target = LazyModel(
                LocalQwenAssets(
                    torch.device("cpu")
                ),
                lambda assets: models.pop()
            )
            _ = DiskPhrasingPairEnvelope(
                PlainBase("Original"),
                ExplicitAttackSetup(
                    QueuedModel([]),
                    target,
                    QueuedModel([]),
                    "Original"
                ),
                root, attempts=1
            ).run()
            process = subprocess.run(
                [
                    sys.executable, "-c", "import json,sys; from pathlib import Path; from journal.parsed_journal import ParsedJournal; root=Path(sys.argv[1]); print(json.dumps([ParsedJournal(root / (name + '.jsonl')).read() for name in ('phrasing_past','phrasing_future')]))",
                    str(root)
                ],
                capture_output=True, text=True, check=True
            )
            assert [(outcome["status"], outcome["reason"], outcome["text"], outcome["calls"]) for outcome in json.loads(process.stdout)] == [
                (
                    "failure", "execution_error", None,
                    {
                        "target": 0,
                        "attacker": 0,
                        "judge": 0
                    }
                )
            ] * 2, "A pre-start model failure skipped future or left unrecoverable journals"

    def test_pair_does_not_overwrite_an_existing_output_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "existing.txt"
            sentinel.write_text(
                "user data", encoding="utf-8"
            )
            with pytest.raises(FileExistsError):
                DiskPhrasingPairEnvelope(
                    PlainBase("Original"),
                    ExplicitAttackSetup(
                        QueuedModel([]),
                        QueuedModel([]),
                        QueuedModel([]),
                        "Original"
                    ),
                    root, attempts=1
                ).run()
            assert sentinel.read_text(encoding="utf-8") == "user data" and len(list(root.iterdir())) == 1, "Phrasing overwrote or modified an existing output directory"

    def test_standalone_displays_both_durable_outcomes_and_returns_failure_for_one_error(self) -> None:
        for replies, code, past in (
            (["Question", "Past", "YES", "Question", "Future", "YES"], 0, "success"),
            (["Question", "Past", "invalid", "Question", "Future", "YES"], 1, "failure"),
            (["", "Question", "Future", "YES"], 1, "failure"),
            (["Question", "Past", "NO", "Question", "Future", "NO"], 0, "round_limit")
        ):
            with TemporaryDirectory() as directory:
                root = Path(directory) / "pair"
                process = subprocess.run(
                    [
                        sys.executable, "-c", f"import main.phrasing_cli as cli; from target.queued_model import QueuedModel; cli.snapshot = lambda offline: None; cli.cuda_device = lambda gpu: 'offline'; cli.LazyQwenModel = lambda device, tokens: QueuedModel({replies!r}); cli.main()",
                        str(root),
                        "Original", "--gpu", "0", "--phrasing-attempts", "1"
                    ],
                    capture_output=True, text=True
                )
                displayed = json.loads(process.stdout)
                assert (
                    process.returncode, displayed["phrasing_past"]["status"], displayed["phrasing_future"]["text"],
                    [{key: value for key, value in displayed[name].items() if key != "error"} for name in ("phrasing_past", "phrasing_future")]
                ) == (
                    code, past, "Future", [ParsedJournal(root / f"{name}.jsonl").read() for name in ("phrasing_past", "phrasing_future")]
                ), "Standalone skipped a tense, changed its durable result or returned the wrong exit status"

    def test_invalid_standalone_settings_fail_before_model_access_or_reservation(self) -> None:
        for gpu, tokens, attempts in ((-1, 192, 20), (0, 0, 20), (0, 192, 0), (0, 192, -1)):
            with TemporaryDirectory() as directory, patch(
                "main.phrasing_cli.LazyQwenModel",
                side_effect=AssertionError("Model accessed")
            ):
                root = Path(directory) / "pair"
                with pytest.raises(ValueError):
                    launch(
                        Namespace(
                            output_dir=root, base="Original", gpu=gpu, tokens=tokens, phrasing_attempts=attempts
                        )
                    )
                assert not root.exists(), "Invalid Phrasing settings accessed the model or reserved output"

    def test_alert_requires_explicit_activation_of_the_two_factories(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with variants(
                root, [], enabled=False, attempts=1
            ) as defaults:
                count = len(defaults)
            with variants(
                root, [], enabled=True, attempts=1
            ) as factories:
                assert (
                    count,
                    len(factories)
                ) == (2, 4), "ALERT changed defaults or omitted one of the two explicitly enabled tenses"

    def test_alert_counts_tense_attempts_instead_of_internal_generations(self) -> None:
        manifest = Path("examples/alert-manifest-640.json")
        bases = AlertPool(manifest).requests(1)[:2]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with variants(
                root,
                [
                    "Question", "Answer", "YES"
                ] * 4, enabled=True, attempts=20
            ) as factories:
                series(
                    bases, factories[-2:], root, manifest, phrasing=True
                )
            report = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
            assert (report["planned"], report["attempted"]) == (4, 4), "ALERT counted internal candidates instead of two independent attempts per request"

    def test_alert_pair_order_survives_a_failed_past_judge(self) -> None:
        manifest = Path("examples/alert-manifest-640.json")
        bases = AlertPool(manifest).requests(1)[:2]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with variants(
                root,
                [
                    "Question", "Past", "invalid", "Question", "Future", "YES", "Question", "Past", "NO", "Question", "Future", "NO"
                ],
                enabled=True, attempts=1
            ) as factories:
                series(
                    bases, factories[-2:], root, manifest, phrasing=True
                )
            report = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
            assert [(entry["base_metadata"]["id"], entry["factory"], entry["outcome"]) for entry in report["runs"]] == [
                (bases[0].identity, 1, "failure"), (bases[0].identity, 2, "completed"),
                (bases[1].identity, 1, "completed"), (bases[1].identity, 2, "completed")
            ], "ALERT skipped future after a failed judge or interleaved different requests within a pair"

    def test_alert_startup_failures_leave_two_independently_recoverable_journals(self) -> None:
        manifest = Path("examples/alert-manifest-640.json")
        bases = AlertPool(manifest).requests(1)[:1]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models: list[Model] = []
            model = LazyModel(
                LocalQwenAssets(
                    torch.device("cpu")
                ),
                lambda assets: models.pop()
            )
            with resources(
                root, model, enabled=True, attempts=1
            ) as factories:
                series(
                    bases, factories[-2:], root, manifest, phrasing=True
                )
            process = subprocess.run(
                [
                    sys.executable, "-c", "import json,sys; from pathlib import Path; from journal.parsed_journal import ParsedJournal; root=Path(sys.argv[1]); print(json.dumps([ParsedJournal(root / name / (sys.argv[2] + '.jsonl')).read() for name in ('phrasing_past','phrasing_future')]))",
                    str(root),
                    str(bases[0].identity)
                ],
                capture_output=True, text=True, check=True
            )
            assert [(outcome["status"], outcome["reason"], outcome["text"]) for outcome in json.loads(process.stdout)] == [
                ("failure", "execution_error", None), ("failure", "execution_error", None)
            ], "ALERT performed metadata loading outside the journaled error boundary or skipped future"

    def test_alert_summary_preserves_the_observed_startup_failure_configuration(self) -> None:
        manifest = Path("examples/alert-manifest-640.json")
        bases = AlertPool(manifest).requests(1)[:1]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models: list[Model] = []
            model = LazyModel(
                LocalQwenAssets(
                    torch.device("cpu")
                ),
                lambda assets: models.pop()
            )
            with resources(
                root, model, enabled=True, attempts=1
            ) as factories:
                series(
                    bases, factories[-2:], root, manifest, phrasing=True
                )
            report = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
            assert [entry["attack"] for entry in report["runs"]] == [
                {
                    "type": "phrasing_past",
                    "description_error": "IndexError('pop from empty list')"
                },
                {
                    "type": "phrasing_future",
                    "description_error": "IndexError('pop from empty list')"
                }
            ], "ALERT retried failed configuration loading instead of preserving the journaled observation"

    def test_alert_keeps_distinct_journals_for_each_tense_and_request(self) -> None:
        manifest = Path("examples/alert-manifest-640.json")
        bases = AlertPool(manifest).requests(1)[:2]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with variants(
                root,
                [
                    "Question", "Past one", "YES", "Question", "Future one", "YES", "Question", "Past two", "YES", "Question", "Future two", "YES"
                ],
                enabled=True, attempts=1
            ) as factories:
                series(
                    bases, factories[-2:], root, manifest, phrasing=True
                )
            assert [ParsedJournal(root / name / f"{base.identity}.jsonl").read()["text"] for base in bases for name in ("phrasing_past", "phrasing_future")] == [
                "Past one", "Future one", "Past two", "Future two"
            ], "ALERT shared or replaced a tense or request journal"

    def test_alert_budget_validation_precedes_output_reservation(self) -> None:
        with TemporaryDirectory() as directory, patch(
            "main.alert_series.LazyQwenModel",
            side_effect=AssertionError("Model accessed")
        ):
            root = Path(directory) / "series"
            with pytest.raises(
                ValueError, match="positive generation budget"
            ):
                alert_launch(
                    Namespace(
                        output_dir=root, gpu=0, tokens=192, turns=1, goat_turns=1, port=8080, phrasing=True, phrasing_attempts=0
                    )
                )
            assert not root.exists(), "Invalid Phrasing budget reserved ALERT output"

    def test_command_parsers_share_budget_defaults_and_alert_opt_in(self) -> None:
        for command, arguments, expected in (
            ("main.phrasing_cli", [], [20, 192, False]),
            ("main.alert_series", [], [20, 192, False]),
            ("main.alert_series", ["--phrasing", "--phrasing-attempts", "7"], [7, 192, True])
        ):
            process = subprocess.run(
                [sys.executable, "-c", f"import json; import {command} as cli; cli.launch = lambda args: (print(json.dumps([args.phrasing_attempts, args.tokens, getattr(args, 'phrasing', False)])), {{}})[1]; cli.main()", "unused", *(["Original"] if command == "main.phrasing_cli" else []), "--gpu", "0", *arguments],
                check=True, capture_output=True, text=True
            )
            assert json.loads(process.stdout.splitlines()[0]) == expected, "Command parser lost the common budget, token limit or explicit ALERT activation"
