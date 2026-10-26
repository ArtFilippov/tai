"""Offline FITD assembly with an independent model conversation per role."""

import tempfile
import unittest
from pathlib import Path

from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from main.alert_series import FITDAttackEnvelope
from ..attack.model_fixture_utils import refusal


class TestFITDAttackEnvelope(unittest.TestCase):
    """Check the bound attack without downloading model weights."""

    def test_bound_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.jsonl"
            model = refusal()
            _ = FITDAttackEnvelope(
                PlainBase("Original request"),
                model, model, model,
                VersionedJournal(
                    DiskJournal(path)
                ),
                1
            ).run()
            assert list(ParsedJournal(path).events())[0]["base"] == "Original request", "The envelope lost its bound request"
