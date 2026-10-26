"""Plain text assembly paths keep their request origin."""

import tempfile
import unittest
from pathlib import Path

from base.plain_base import PlainBase
from journal.parsed_journal import ParsedJournal
from main.alert_series import assemble
from main.qwen_observe import observe
from .alert_fixture_utils import refusal


class TestQwenRun(unittest.TestCase):
    """Check offline Qwen assembly without model weights."""

    def test_observation(self) -> None:
        journal = observe(
            "Free-form request",
            refusal(),
            1
        )
        assert journal.events()[0] == {
            "event": "start",
            "base": "Free-form request",
            "base_metadata": {}
        }, "The observation did not preserve a plain text request"

    def test_qwen_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            assemble(
                refusal(),
                refusal(),
                1
            ).run(
                PlainBase("Free-form request"),
                path
            )
            assert list(ParsedJournal(path).events())[0]["base_metadata"] == {}, "The Qwen run invented source taxonomy"
