"""Functional dataset results remain available as journal events."""

import unittest

from journal.memory_journal import MemoryJournal
from journal.recorded_functional_test import RecordedFunctionalTest
from journal.versioned_journal import VersionedJournal


class TestRecordedFunctionalTest(unittest.TestCase):
    """Check recorded metadata and benchmark results without a benchmark runner."""

    def test_records_all_datasets(self) -> None:
        memory = MemoryJournal()
        test = RecordedFunctionalTest(
            VersionedJournal(memory)
        )
        test.start("local")
        test.record(
            "ruMMLU", {"acc": 0.25}, 12.5
        )
        test.record(
            "HumanEval", {"pass_at_1": 1}, 3.0
        )
        assert memory.events() == [
            {
                "event": "functional_test",
                "logs_type": "generative_models_functional_test",
                "mlops": "local",
                "schema_id": "tai.memory",
                "schema_version": 1
            },
            {
                "event": "dataset",
                "task_name": "ruMMLU",
                "results": {"acc": 0.25},
                "time": 12.5,
                "schema_id": "tai.memory",
                "schema_version": 1
            },
            {
                "event": "dataset",
                "task_name": "HumanEval",
                "results": {"pass_at_1": 1},
                "time": 3.0,
                "schema_id": "tai.memory",
                "schema_version": 1
            }
        ], "A functional result was not recorded with its experiment and schema"
