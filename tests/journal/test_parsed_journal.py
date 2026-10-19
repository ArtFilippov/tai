"""Recovery of a run outcome from a saved journal."""

import subprocess
import sys
import unittest
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from ..journal_utils import temporary_journal


class TestParsedJournal(unittest.TestCase):
    """Checks lossless result and assessment recovery across process boundaries."""

    def file(self, name: str) -> AbstractContextManager[Path]:
        return temporary_journal(name)

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

    def test_no_result(self) -> None:
        with self.file("none.jsonl") as path:
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

    def test_empty_result(self) -> None:
        with self.file("empty.jsonl") as path:
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

    def test_negative_assessment(self) -> None:
        with self.file("negative.jsonl") as path:
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

    def test_latest_linked_assessment(self) -> None:
        with self.file("assessments.jsonl") as path:
            self.save(
                path,
                [
                    {
                        "event": "result",
                        "text": "First"
                    },
                    {
                        "event": "judgment",
                        "text": "First",
                        "assessment": False
                    },
                    {
                        "event": "result",
                        "text": "Second"
                    },
                    {
                        "event": "judgment",
                        "text": "Second",
                        "assessment": True
                    },
                    {
                        "event": "end",
                        "status": "success"
                    }
                ]
            )
            assert ParsedJournal(path).read()["text"] == "Second", "The latest assessment was linked to the wrong result"

    def test_rejects_false_success(self) -> None:
        with self.file("invalid-success.jsonl") as path:
            self.save(
                path,
                [
                    {
                        "event": "result",
                        "text": "Unsuccessful"
                    },
                    {
                        "event": "judgment",
                        "text": "Unsuccessful",
                        "assessment": False
                    },
                    {
                        "event": "end",
                        "status": "success"
                    }
                ]
            )
            with pytest.raises(
                ValueError, match="Invalid"
            ):
                ParsedJournal(path).read()

    def test_absent_assessment_is_incomplete(self) -> None:
        with self.file("incomplete.jsonl") as path:
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

    def test_rejects_unlinked_judgment(self) -> None:
        with self.file("mismatch.jsonl") as path:
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

    def test_new_process(self) -> None:
        with self.file("process.jsonl") as path:
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
                    sys.executable, "-c", "from journal.parsed_journal import ParsedJournal; from pathlib import Path; import sys; print(repr(ParsedJournal(Path(sys.argv[1])).read()))",
                    str(path)
                ],
                check=True, capture_output=True, text=True
            )
            assert "'text': 'Exact\\ntext', 'assessment': False" in process.stdout, "A new process did not recover the exact judgment"
