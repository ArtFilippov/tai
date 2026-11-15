"""Target-bound rendering of the pinned DAN catalog."""

from typing import Protocol


class DanText(Protocol):
    """Render native activation text and identify its pinned source."""

    def render(self, text: str) -> str:
        ...

    def description(self) -> dict[str, object]:
        ...
