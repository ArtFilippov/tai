"""Named DAN entry points with real disk journals and offline participants."""

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

from attack.dan.envelopes.dan_attack_envelope import DanAttackEnvelope
from attack.explicit_attack_setup import ExplicitAttackSetup
from base.alert_pool import AlertPool
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from main.alert_series import series
from main.dan_cli import launch
from ....journal_utils import temporary_journal
from ....target.test_model import TestModel as OfflineModel
from ..dan_entry_utils import variants


class TestDanAttackEnvelope(unittest.TestCase):
    """Expose one request-bound attempt and the recorded first successful source."""

    def attempt(self, path: Path, answer: str, verdict: str) -> str | None:
        """Run the public assembly with an exact request and controlled model.

        Args:
            path (Path): Independent journal destination.
            answer (str): Full response for both Target exchanges.
            verdict (str): Independent fulfillment verdict.

        Returns:
            str | None: Presented Target text.
        """
        model = OfflineModel(
            answer, verdict, name="Offline/Target"
        )
        return DanAttackEnvelope(
            PlainBase("  Explain café\nexactly  "),
            ExplicitAttackSetup(
                model, model, model, "  Explain café\nexactly  "
            ),
            VersionedJournal(
                DiskJournal(path)
            ),
            sections=["named"]
        ).run()

    def test_python_assembly_recovers_first_success_linked_to_target(self) -> None:
        with temporary_journal("named.jsonl") as path:
            self.attempt(
                path, "Complete answer", "YES"
            )
            outcome = ParsedJournal(path).read()
            assert (outcome["successful_candidate"], outcome["target"]) == (
                {
                    "section": "named",
                    "variant": "Dan_11_0",
                    "source_index": 0,
                    "position": 1
                },
                {
                    "type": "model_chat",
                    "model": {
                        "id": "Offline/Target",
                        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                    },
                },
            ), "Python DAN wiring lost its recorded first-success source or Target identity"

    def test_python_assembly_returns_and_assesses_the_complete_second_reply(self) -> None:
        with temporary_journal("full-reply.jsonl") as path:
            answer = "[GPT] No.\n[DAN] Complete café answer.\nTrailing text"
            result = self.attempt(
                path, answer, "YES"
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert (result, [event["prompt"] for event in events if event.get("role") == "judge"]) == (
                answer, [answer],
            ), "Python DAN assembly extracted a persona fragment instead of assessing the full reply"

    def test_python_assembly_preserves_the_exact_base_request(self) -> None:
        with temporary_journal("exact-base.jsonl") as path:
            self.attempt(
                path, "Complete answer", "YES"
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert [event["prompt"] for event in events if event.get("phase") == "base_request"] == [
                "  Explain café\nexactly  "
            ], "The live DAN assembly changed the base request"

    def test_static_configuration_identifies_named_section_and_pinned_source(self) -> None:
        with temporary_journal("configuration.jsonl") as path:
            self.attempt(
                path, "Complete answer", "YES"
            )
            start = json.loads(
                path.read_text(encoding="utf-8")
                .splitlines()[0]
            )
            assert start["attack"]["probe"]["collection"] == {
                "type": "named_dan_prompts",
                "sections": ["named"],
                "source": "garak.probes.dan",
                "revision": "3f50ea5ff9cd7050099940647c15c39b07a93392",
            }, "The recorded static DAN description lost its selected section or pinned revision"

    def test_persona_acknowledgment_does_not_count_as_fulfillment(self) -> None:
        with temporary_journal("exhausted.jsonl") as path:
            self.attempt(
                path, "DAN mode enabled", "NO"
            )
            outcome = ParsedJournal(path).read()
            assert (
                outcome["reason"], outcome["text"], outcome["assessment"],
                outcome.get("successful_candidate"),
                outcome["calls"]
            ) == (
                "candidates_exhausted", "DAN mode enabled", False, None,
                {
                    "target": 28,
                    "attacker": 0,
                    "judge": 14
                }
            ), "Persona markers became success or named candidates became separate attempts"

    def test_python_assembly_rejects_unavailable_and_empty_sections(self) -> None:
        for selection in ([], ["wild"], ["ablation"], ["named", "wild"]):
            with temporary_journal("invalid.jsonl") as path, pytest.raises(
                ValueError, match="only 'named'"
            ):
                model = OfflineModel(
                    "answer", "YES", name="Offline/Target"
                )
                DanAttackEnvelope(
                    PlainBase("base"),
                    ExplicitAttackSetup(
                        model, model, model, "base"
                    ),
                    VersionedJournal(
                        DiskJournal(path)
                    ),
                    sections=selection
                )

    def test_malformed_fulfillment_verdict_records_error_without_winner(self) -> None:
        with temporary_journal("judge-error.jsonl") as path:
            with pytest.raises(
                ValueError, match="Malformed judge verdict"
            ):
                self.attempt(
                    path, "DAN mode enabled", "invalid"
                )
            outcome = ParsedJournal(path).read()
            assert (
                outcome["status"], outcome["reason"], outcome["text"],
                outcome.get("successful_candidate")
            ) == (
                "failure", "assessment_error", "DAN mode enabled", None,
            ), "An invalid fulfillment verdict erased the reply or claimed a successful variant"

    def test_standalone_recovers_first_success_with_default_and_explicit_section(self) -> None:
        for selection in ([], ["--dan-sections", "named"]):
            with temporary_journal("cli.jsonl") as path:
                script = (
                    "import main.dan_cli as cli; "
                    "from tests.target.test_model import TestModel; "
                    "cli.snapshot = lambda offline: None; "
                    "cli.cuda_device = lambda gpu: 'cuda:' + str(gpu); "
                    "cli.LazyQwenModel = lambda device, tokens: TestModel('Complete answer', 'YES', name=f'CLI/{device}/{tokens}'); "
                    "cli.main()"
                )
                result = subprocess.run(
                    [
                        sys.executable, "-c", script,
                        str(path),
                        "Base café", "--gpu", "3", "--tokens", "73", *selection
                    ],
                    check=True, capture_output=True, text=True
                )
                report = ast.literal_eval(result.stdout)
                assert report == ParsedJournal(path).read() and (
                    report["successful_candidate"]["variant"], report["target"]["model"]["id"]
                ) == ("Dan_11_0", "CLI/cuda:3/73"), "Standalone output lost its first-success identity or GPU/token setup"

    def test_command_validation_precedes_model_access_and_reservation(self) -> None:
        for gpu, tokens, selection in ((-1, 8, ["named"]), (0, 0, ["named"]), (0, 8, []), (0, 8, ["wild"])):
            with temporary_journal("invalid-cli.jsonl") as path, patch(
                "main.dan_cli.LazyQwenModel", side_effect=AssertionError("Model accessed")
            ):
                with pytest.raises(ValueError):
                    launch(
                        Namespace(
                            journal=path, base="base", gpu=gpu, tokens=tokens, dan_sections=selection
                        )
                    )
                assert not path.exists(), "Invalid DAN arguments reserved a journal"

    def test_cli_parser_rejects_empty_and_unavailable_sections(self) -> None:
        for command in (["-m", "main.dan_cli", "unused.jsonl", "base"], ["-c", "from main.alert_series import main; main()", "unused-directory"]):
            for selection in (["--dan-sections"], ["--dan-sections", "wild"]):
                result = subprocess.run(
                    [sys.executable, *command, "--gpu", "0", *selection],
                    capture_output=True, text=True,
                )
                assert result.returncode == 2, "An entry point accepted an unavailable or empty section selection"

    def test_alert_parser_exposes_named_selection_without_enabling_it_by_default(self) -> None:
        for selection, expected in (([], [False, ["named"]]), (["--dan", "--dan-sections", "named"], [True, ["named"]])):
            result = subprocess.run(
                [
                    sys.executable, "-c", "import json; import main.alert_series as cli; " "cli.launch = lambda args: print(json.dumps([args.dan, args.dan_sections])); cli.main()", "unused-directory", "--gpu", "0", *selection
                ],
                check=True, capture_output=True, text=True
            )
            assert json.loads(result.stdout) == expected, "ALERT changed default variants or lost named-section selection"

    def test_alert_dan_variant_requires_explicit_opt_in(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with variants(
                root, enabled=False
            ) as defaults:
                count = len(defaults)
            with variants(
                root, enabled=True
            ) as selected:
                assert (
                    count,
                    len(selected)
                ) == (2, 3), "DAN changed default variants or failed to add exactly one opt-in variant"

    def test_alert_counts_one_attempt_per_base_instead_of_per_candidate(self) -> None:
        manifest = Path("examples/alert-manifest-640.json")
        bases = AlertPool(manifest).requests(1)[:2]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with variants(
                root, enabled=True
            ) as factories:
                series(
                    bases, factories[-1:], root, manifest
                )
            report = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
            assert (report["planned"], report["attempted"]) == (2, 2), "Internal DAN candidates inflated root series attempts"

    def test_alert_preserves_independent_journals_for_each_base(self) -> None:
        manifest = Path("examples/alert-manifest-640.json")
        bases = AlertPool(manifest).requests(1)[:2]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with variants(
                root, enabled=True
            ) as factories:
                series(
                    bases, factories[-1:], root, manifest
                )
            outcomes = [ParsedJournal(root / "dan" / f"{base.identity}.jsonl").read() for base in bases]
            assert [outcome["calls"] for outcome in outcomes] == [
                    {
                        "target": 28,
                        "attacker": 0,
                        "judge": 14
                    }
                ] * 2, "DAN journals shared calls or lost candidates from an independent base request"
