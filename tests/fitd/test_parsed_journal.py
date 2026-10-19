"""Recovery of a run outcome from a saved journal."""

import subprocess
import sys
from pathlib import Path

import pytest

from fitd.disk_journal import DiskJournal
from fitd.parsed_journal import ParsedJournal


class TestParsedJournal:
    """Checks lossless result and assessment recovery across process boundaries."""

    def save(self, path: Path, events: list[dict[str, object]]) -> None:
        journal = DiskJournal(path)
        journal.record(
            {
                "event": "start",
                "base": "Independent request"
            }
        )
        for event in events:
            journal.record(event)

    def test_no_result(self, tmp_path: Path) -> None:
        path = tmp_path / "none.jsonl"
        self.save(
            path,
            [
                {
                    "event": "end",
                    "status": "no_result"
                }
            ]
        )
        assert ParsedJournal(path).read()["text"] is None, "An absent result became text"

    def test_empty_result(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        self.save(
            path,
            [
                {
                    "event": "result",
                    "text": ""
                },
                {
                    "event": "judgment",
                    "text": "",
                    "assessment": False
                },
                {
                    "event": "end",
                    "status": "completed"
                }
            ]
        )
        text = ParsedJournal(path).read()["text"]
        assert text is not None and not text, "An empty result was lost"

    def test_negative_assessment(self, tmp_path: Path) -> None:
        path = tmp_path / "negative.jsonl"
        self.save(
            path,
            [
                {
                    "event": "result",
                    "text": "Result"
                },
                {
                    "event": "judgment",
                    "text": "Result",
                    "assessment": False
                },
                {
                    "event": "end",
                    "status": "completed"
                }
            ]
        )
        assert ParsedJournal(path).read()["assessment"] is False, "A negative verdict was lost"

    def test_absent_assessment_is_incomplete(self, tmp_path: Path) -> None:
        path = tmp_path / "incomplete.jsonl"
        self.save(
            path,
            [
                {
                    "event": "result",
                    "text": "Result"
                }
            ]
        )
        with pytest.raises(
            ValueError, match="no outcome"
        ):
            ParsedJournal(path).read()

    def test_rejects_unlinked_judgment(self, tmp_path: Path) -> None:
        path = tmp_path / "mismatch.jsonl"
        self.save(
            path,
            [
                {
                    "event": "result",
                    "text": "Result"
                },
                {
                    "event": "judgment",
                    "text": "Other",
                    "assessment": True
                }
            ]
        )
        with pytest.raises(
            ValueError, match="Invalid"
        ):
            ParsedJournal(path).read()

    def test_new_process(self, tmp_path: Path) -> None:
        path = tmp_path / "process.jsonl"
        self.save(
            path,
            [
                {
                    "event": "result",
                    "text": "Exact\ntext"
                },
                {
                    "event": "judgment",
                    "text": "Exact\ntext",
                    "assessment": False
                },
                {
                    "event": "end",
                    "status": "completed"
                }
            ]
        )
        process = subprocess.run(
            [
                sys.executable, "-c", "from fitd.parsed_journal import ParsedJournal; from pathlib import Path; import sys; print(repr(ParsedJournal(Path(sys.argv[1])).read()))",
                str(path)
            ],
            check=True, capture_output=True, text=True
        )
        assert "'text': 'Exact\\ntext', 'assessment': False" in process.stdout, "A new process did not recover the exact judgment"
