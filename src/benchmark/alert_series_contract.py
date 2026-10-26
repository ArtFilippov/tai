"""A series index assembled from available run information."""

from typing import Protocol, Any


class AlertSeriesContract(Protocol):
    """Save the planned and attempted attacks without inferring assessments."""

    def report(self) -> dict[str, Any]: ...

    def save(self) -> None: ...

    def record(self, entry: dict[str, object]) -> None: ...
