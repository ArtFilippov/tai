"""An independently budgeted tense-reformulation experiment."""

from typing import Protocol


class PhrasingPlan(Protocol):
    """Own a tense's identity, pinned instruction and bounded generation sequence."""

    def kind(self) -> str:
        ...

    def generations(self) -> range:
        ...

    def instruction(self, request: str) -> str:
        ...

    def description(self) -> dict[str, object]:
        ...
