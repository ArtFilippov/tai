"""Observable results of a functional generative-model evaluation."""

from collections.abc import Mapping
from typing import Protocol


class FunctionalTest(Protocol):
    """Record experiment metadata and one result per dataset."""

    def start(self, mlops: str) -> None:
        ...

    def record(self, task_name: str, results: Mapping[str, int | float], seconds: float) -> None:
        ...
