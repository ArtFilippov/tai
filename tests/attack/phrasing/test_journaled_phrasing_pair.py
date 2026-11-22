"""Preserve original failures when the read-side journal also fails."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from attack.explicit_attack_setup import ExplicitAttackSetup
from attack.phrasing.configured_phrasing_plan import ConfiguredPhrasingPlan
from attack.phrasing.envelopes.phrasing_attack_envelope import PhrasingAttackEnvelope
from attack.phrasing.journaled_phrasing_pair import JournaledPhrasingPair
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from target.queued_model import QueuedModel


class TestJournaledPhrasingPair(unittest.TestCase):
    """Keep both execution and recovery error contexts at the pair boundary."""

    def test_failed_recovery_retains_its_context_and_attempts_the_next_tense(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base = PlainBase("Original")
            setup = ExplicitAttackSetup(
                QueuedModel(["Question", "Question"]),
                QueuedModel(["Past", "Future"]),
                QueuedModel(["invalid", "invalid"]),
                "Original"
            )
            outcomes = JournaledPhrasingPair(
                root / "pair",
                [
                    ConfiguredPhrasingPlan(
                        "past", 1
                    ),
                    ConfiguredPhrasingPlan(
                        "future", 1
                    )
                ],
                lambda plan, path: PhrasingAttackEnvelope(base, setup, VersionedJournal(DiskJournal(path)), plan=plan), lambda path: ParsedJournal(root / "missing.jsonl")
            ).run()
            assert [(outcomes[name]["status"], "Malformed judge verdict" in str(outcomes[name]["error"]), "FileNotFoundError" in str(outcomes[name]["recovery_error"]), outcomes[name]["journal"]) for name in ("phrasing_past", "phrasing_future")] == [
                (
                    "failure", True, True,
                    str(root / "pair" / "phrasing_past.jsonl")
                ),
                (
                    "failure", True, True,
                    str(root / "pair" / "phrasing_future.jsonl")
                )
            ], "Journal recovery hid the original failure, lost its own context or skipped future"
