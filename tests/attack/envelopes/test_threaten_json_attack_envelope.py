"""Bound ThreatenJSON retains its distinct JSON answer assessment."""

import tempfile
import unittest
from pathlib import Path

from attack.shared_model_attack_setup import SharedModelAttackSetup
from attack.envelopes.threaten_json_attack_envelope import ThreatenJSONAttackEnvelope
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from ..model_fixture_utils import refusal


class TestThreatenJSONAttackEnvelope(unittest.TestCase):
    """Check the bound run retains its JSON-specific assessment."""

    def test_non_json_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.jsonl"
            _ = ThreatenJSONAttackEnvelope(
                PlainBase("Original request"),
                SharedModelAttackSetup(
                    refusal(), "Original request"
                ),
                VersionedJournal(
                    DiskJournal(path)
                )
            ).run()
            assert ParsedJournal(path).read()["assessment"] is False, "The bound JSON judge accepted a non-JSON result"
