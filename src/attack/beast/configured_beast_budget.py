"""Named BEAST search dimensions with garak defaults."""

from dataclasses import dataclass
from typing import override

from attack.beast.beast_budget import BeastBudget


@dataclass(frozen=True)
class ConfiguredBeastBudget(BeastBudget):
    """Keep beams, samples, per-trial length and trials together.

    Attributes:
        width (int): Retained beams.
        count (int): Sampled continuations per beam.
        length (int): Tokens added in one trial.
        repeats (int): Continued trials per request.
    """

    width: int
    count: int
    length: int
    repeats: int

    @override
    def beams(self) -> int:
        return self.width

    @override
    def samples(self) -> int:
        return self.count

    @override
    def suffix_len(self) -> int:
        return self.length

    @override
    def trials(self) -> int:
        return self.repeats

    @override
    def validate(self) -> None:
        """Reject nonpositive search dimensions.

        Raises:
            ValueError: If the search cannot be bounded.
        """
        if min(self.width, self.count, self.length, self.repeats) < 1:
            raise ValueError("BEAST search limits must be positive")
