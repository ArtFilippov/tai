"""Pinned research instructions for one bounded past/future experiment."""

from importlib.resources import files
from typing import override

from attack.phrasing.phrasing_plan import PhrasingPlan


class ConfiguredPhrasingPlan(PhrasingPlan):
    """Keep the original instruction and Attacker-call budget together."""

    def __init__(self, tense: str, attempts: int) -> None:
        """Bind a tense and a finite preparation budget.

        Args:
            tense (str): Past or future.
            attempts (int): Maximum preparation calls, including duplicates.
        """
        super().__init__()
        self.tense = tense
        self.attempts = attempts

    @override
    def kind(self) -> str:
        return self.tense

    @override
    def generations(self) -> range:
        """Enumerate preparation calls after validating their finite bound.

        Returns:
            range: One-based generation positions.

        Raises:
            ValueError: If the tense or budget is invalid.
        """
        if self.tense not in ("past", "future") or self.attempts < 1:
            raise ValueError("Phrasing requires past/future and a positive generation budget")
        return range(
            1, self.attempts + 1
        )

    @override
    def instruction(self, request: str) -> str:
        _ = self.generations()
        return (
            files("attack.phrasing").joinpath(f"{self.tense}.txt")
            .read_text(encoding="utf-8")
            .format(request=request)
        )

    @override
    def description(self) -> dict[str, object]:
        _ = self.generations()
        return {
            "tense": self.tense,
            "attempts": self.attempts,
            "template": files("attack.phrasing").joinpath(f"{self.tense}.txt")
            .read_text(encoding="utf-8"),
            "source": "garak.probes.phrasing",
            "revision": "3f50ea5ff9cd7050099940647c15c39b07a93392",
            "preparation_source": "https://github.com/tml-epfl/llm-past-tense/blob/179e305ddeecbfbd4d7c05fab626faf9b2ac5f44/main.py"
        }
