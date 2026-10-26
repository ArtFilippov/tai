"""Model configuration in a start event must preserve outcome recovery."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from journal.configured_journal import ConfiguredJournal
from journal.disk_journal import DiskJournal
from journal.memory_journal import MemoryJournal
from journal.parsed_journal import ParsedJournal
from target.scripted_model import ScriptedModel


class TestConfiguredJournal(unittest.TestCase):
    """Check that extra model metadata remains a valid run event."""

    def test_includes_selected_model_in_start_event(self) -> None:
        memory = MemoryJournal()
        journal = ConfiguredJournal(
            memory,
            ScriptedModel([]),
            ScriptedModel([])
        )
        journal.record(
            {
                "event": "start",
                "base": "Custom request"
            }
        )
        assert memory.events()[0]["models"] == {
            "target": {
                "id": "scripted",
                "revision": "scripted"
            },
            "judge": {
                "id": "scripted",
                "revision": "scripted"
            },
            "attacker": {
                "id": "scripted",
                "revision": "scripted"
            }
        }, "The start event omitted a model's settings"

    def test_recovers_outcome_with_model_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            journal = ConfiguredJournal(
                DiskJournal(path),
                ScriptedModel([]),
                ScriptedModel([])
            )
            journal.record(
                {
                    "event": "start",
                    "base": "Custom request"
                }
            )
            journal.record(
                {
                    "event": "end",
                    "status": "no_result",
                    "reason": "no_result"
                }
            )
            assert ParsedJournal(path).read()["status"] == "no_result", "Model metadata broke journal recovery"
