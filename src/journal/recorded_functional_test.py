"""Persist supplied functional results without executing benchmarks."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import override

from journal.functional_test import FunctionalTest
from journal.journal import Journal


@dataclass(eq=False)
class RecordedFunctionalTest(FunctionalTest):
    """Send experiment and dataset facts to an injected run journal.

    Attributes:
        journal (Journal): Destination for the recorded facts.
    """

    journal: Journal

    @override
    def start(self, mlops: str) -> None:
        self.journal.record(
            {
                "event": "functional_test",
                "logs_type": "generative_models_functional_test",
                "mlops": mlops
            }
        )

    @override
    def record(self, task_name: str, results: Mapping[str, int | float], seconds: float) -> None:
        self.journal.record(
            {
                "event": "dataset",
                "task_name": task_name,
                "results": dict(results),
                "time": seconds
            }
        )
