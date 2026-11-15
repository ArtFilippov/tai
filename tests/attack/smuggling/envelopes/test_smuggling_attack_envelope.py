"""Python and command entry points with scripted resources and real journals."""

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

from attack.explicit_attack_setup import ExplicitAttackSetup
from attack.smuggling.envelopes.smuggling_attack_envelope import SmugglingAttackEnvelope
from base.plain_base import PlainBase
from base.alert_pool import AlertPool
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.queued_model import QueuedModel
from main.alert_series import series, launch as alert_launch
from main.smuggling_cli import launch
from ....journal_utils import temporary_journal
from ..smuggling_entry_utils import standalone, variants


class TestSmugglingAttackEnvelope(unittest.TestCase):
    """Expose one request-bound attack with the normal fulfillment Judge."""

    def test_python_assembly_assesses_one_complete_response_without_attacker_calls(self) -> None:
        with temporary_journal("python.jsonl") as path:
            result = SmugglingAttackEnvelope(
                PlainBase("Explain café"),
                ExplicitAttackSetup(
                    QueuedModel([]),
                    QueuedModel(["Full answer\nTrailing text"]),
                    QueuedModel(["YES"]),
                    "Explain café"
                ),
                VersionedJournal(
                    DiskJournal(path)
                )
            ).run()
            outcome = ParsedJournal(path).read()
            assert (result, outcome["assessment"], outcome["successful_candidate"], outcome["calls"]) == (
                "Full answer\nTrailing text", True,
                {
                    "technique": "homoglyph",
                    "position": 1,
                    "seed": 0
                },
                {
                    "target": 1,
                    "attacker": 0,
                    "judge": 1
                }
            ), "Python assembly lost its response, fulfillment verdict, or technique, or made extra calls"

    def test_standalone_default_and_explicit_selection_return_recorded_outcome(self) -> None:
        for selection in ([], ["--smuggling-techniques", "homoglyph"]):
            with temporary_journal("standalone.jsonl") as path:
                script = (
                    "import main.smuggling_cli as cli; from target.queued_model import QueuedModel; "
                    "cli.snapshot = lambda offline: None; cli.cuda_device = lambda gpu: 'offline'; "
                    "cli.LazyQwenModel = lambda device, tokens: QueuedModel(['Complete answer', 'YES']); cli.main()"
                )
                process = subprocess.run(
                    [
                        sys.executable, "-c", script,
                        str(path),
                        "Base café", "--gpu", "0", *selection
                    ],
                    check=True, capture_output=True, text=True
                )
                outcome = ast.literal_eval(process.stdout)
                assert (outcome, outcome["successful_candidate"]["technique"]) == (
                    ParsedJournal(path).read(),
                    "homoglyph"
                ), "Standalone selection failed to execute or display the recoverable Homoglyph outcome"

    def test_alert_adds_one_opt_in_variant_and_gives_each_base_an_independent_journal(self) -> None:
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
                    "First answer", "NO", "Second answer", "YES"
                ],
                enabled=True
            ) as factories:
                selected = len(factories)
                series(
                    bases, factories[-1:], root, manifest
                )
            outcomes = [ParsedJournal(root / "smuggling" / f"{base.identity}.jsonl").read() for base in bases]
            report = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
            assert (
                count, selected, report["planned"], report["attempted"],
                [(outcome["text"], outcome["assessment"], outcome["calls"]) for outcome in outcomes]
            ) == (
                2, 3, 2, 2,
                [
                    (
                        "First answer", False,
                        {
                            "target": 1,
                            "attacker": 0,
                            "judge": 1
                        }
                    ),
                    (
                        "Second answer", True,
                        {
                            "target": 1,
                            "attacker": 0,
                            "judge": 1
                        }
                    )
                ]
            ), "ALERT changed default variants, counted internal candidates, or shared Smuggling journals between requests"

    def test_python_selection_rejects_empty_and_unsupported_techniques(self) -> None:
        for selection in ([], ["unknown"], ["hypothetical"], ["homoglyph", "unknown"]):
            with temporary_journal("invalid-python.jsonl") as path, pytest.raises(
                ValueError, match="nonempty subset"
            ):
                SmugglingAttackEnvelope(
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

    def test_duplicate_selection_still_generates_exactly_one_candidate(self) -> None:
        with temporary_journal("duplicate.jsonl") as path:
            SmugglingAttackEnvelope(
                PlainBase("Base"),
                ExplicitAttackSetup(
                    QueuedModel([]),
                    QueuedModel(["Answer"]),
                    QueuedModel(["NO"]),
                    "Base"
                ),
                VersionedJournal(
                    DiskJournal(path)
                ),
                techniques=[
                    "homoglyph", "homoglyph"
                ]
            ).run()
            assert ParsedJournal(path).read()["calls"] == {
                "target": 1,
                "attacker": 0,
                "judge": 1
            }, "Duplicate selection generated additional candidates"

    def test_standalone_validation_precedes_model_access_and_journal_reservation(self) -> None:
        for gpu, tokens, selection in ((-1, 8, ["homoglyph"]), (0, 0, ["homoglyph"]), (0, 8, []), (0, 8, ["hypothetical"])):
            with temporary_journal("invalid-cli.jsonl") as path, patch(
                "main.smuggling_cli.LazyQwenModel", side_effect=AssertionError("Model accessed")
            ):
                with pytest.raises(ValueError):
                    launch(
                        Namespace(
                            journal=path, base="Base", gpu=gpu, tokens=tokens, smuggling_techniques=selection
                        )
                    )
                assert not path.exists(), "Invalid standalone arguments reserved a journal"

    def test_alert_validation_precedes_model_access_and_output_reservation(self) -> None:
        for selection in ([], ["unknown"], ["homoglyph", "hypothetical"]):
            with TemporaryDirectory() as directory, patch(
                "main.alert_series.LazyQwenModel", side_effect=AssertionError("Model accessed")
            ):
                output = Path(directory) / "output"
                with pytest.raises(
                    ValueError, match="nonempty subset"
                ):
                    alert_launch(
                        Namespace(
                            output_dir=output, gpu=0, tokens=8, turns=1, goat_turns=1, port=8080, smuggling=True, smuggling_techniques=selection
                        )
                    )
                assert not output.exists(), "Invalid ALERT selection reserved output"

    def test_both_parsers_reject_empty_and_unsupported_selections(self) -> None:
        for command in (["-m", "main.smuggling_cli", "unused.jsonl", "base"],
                        ["-c", "from main.alert_series import main; main()", "unused-directory"]):
            for selection in (["--smuggling-techniques"], ["--smuggling-techniques", "hypothetical"]):
                process = subprocess.run(
                    [sys.executable, *command, "--gpu", "0", *selection], capture_output=True, text=True
                )
                assert process.returncode == 2, "A command parser accepted an empty or unsupported Smuggling selection"

    def test_alert_parser_defaults_to_disabled_smuggling_and_preserves_explicit_selection(self) -> None:
        for selection, expected in (([], [False, ["homoglyph"]]),
                                    (["--smuggling", "--smuggling-techniques", "homoglyph"], [True, ["homoglyph"]])):
            process = subprocess.run(
                [
                    sys.executable, "-c", "import json; import main.alert_series as cli; " "cli.launch = lambda args: print(json.dumps([args.smuggling, args.smuggling_techniques])); cli.main()", "unused-directory", "--gpu", "0", *selection
                ],
                check=True, capture_output=True, text=True
            )
            assert json.loads(process.stdout) == expected, "ALERT enabled Smuggling by default or lost explicit technique selection"

    def test_installed_standalone_command_exposes_technique_selection(self) -> None:
        process = subprocess.run(
            [
                "tai-smuggling", "--help"
            ],
            check=True, capture_output=True, text=True
        )
        assert "--smuggling-techniques {homoglyph}" in process.stdout, "The installed tai-smuggling entry point lost technique selection"

    def test_standalone_unchanged_request_has_no_result_or_model_calls(self) -> None:
        with temporary_journal("cli-skip.jsonl") as path:
            standalone(
                path, "123 Ж🙂", []
            )
            outcome = ParsedJournal(path).read()
            assert (outcome["text"], outcome["status"], outcome["assessment"], outcome["calls"]) == (
                None, "no_result", None,
                {
                    "target": 0,
                    "attacker": 0,
                    "judge": 0
                }
            ), "Standalone sent an unchanged request or confused absence with a negative assessment"

    def test_standalone_errors_propagate_and_preserve_the_failed_run(self) -> None:
        for replies, text, reason in (([], None, "execution_error"), (["Selected", "invalid"], "Selected", "assessment_error")):
            with temporary_journal("cli-error.jsonl") as path:
                with pytest.raises(ValueError):
                    standalone(
                        path, "Base café", replies
                    )
                outcome = ParsedJournal(path).read()
                assert (outcome["status"], outcome["reason"], outcome["text"], outcome["assessment"]) == (
                    "failure", reason, text, None
                ), "Standalone converted a technical error into exhaustion or lost the presented result"

    def test_alert_continues_after_failed_smuggling_attempt_with_an_independent_journal(self) -> None:
        manifest = Path("examples/alert-manifest-640.json")
        bases = AlertPool(manifest).requests(1)[:2]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with variants(
                root,
                [
                    "Selected", "invalid", "Complete answer", "YES"
                ],
                enabled=True
            ) as factories:
                series(
                    bases, factories[-1:], root, manifest
                )
            outcomes = [ParsedJournal(root / "smuggling" / f"{base.identity}.jsonl").read() for base in bases]
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
                        "technique": "homoglyph",
                        "position": 1,
                        "seed": 0
                    }
                )
            ], "ALERT aborted later attempts or carried the failed attempt's outcome into another journal"
