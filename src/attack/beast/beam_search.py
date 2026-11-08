"""BEAST beam expansion with continued trials and loss-based selection."""

import math
from typing import override

import torch

from attack.beast.beast_budget import BeastBudget
from attack.beast.beast_search import BeastSearch
from journal.journal import Journal
from target.token_probabilities import TokenProbabilities


class BeamSearch(BeastSearch):
    """Keep the best scored suffix across all depths and continued trials."""

    def __init__(self, tokens: TokenProbabilities, budget: BeastBudget) -> None:
        """Bind local token operations and one attempt's events.

        Args:
            tokens (TokenProbabilities): Compatible distributions and losses.
            budget (BeastBudget): Finite per-request search bounds.
        """
        super().__init__()
        self.tokens = tokens
        self.budget = budget
        self.best: float = math.inf

    @override
    def sample(self, base: str, suffixes: list[list[int]], count: int) -> list[list[int]]:
        """Choose distinct continuations from each user-prefix distribution.

        Args:
            base (str): Original user request.
            suffixes (list[list[int]]): Token sequences of the current beams.
            count (int): Number of continuations per beam.

        Returns:
            list[list[int]]: Extended beams in the model's token space.
        """
        return [
            suffix + [
                int(
                    token.item()
                )
            ]
            for suffix, distribution in zip(
                suffixes,
                self.tokens.distribution(
                    [base + self.tokens.decode(suffix) for suffix in suffixes]
                ), strict=True
            )
            for token in (
                torch.multinomial(distribution, count, replacement=False)
                if distribution.numel() else torch.empty(0, dtype=torch.long, device=distribution.device)
            )
        ]

    @override
    def description(self) -> dict[str, object]:
        return {
            "scorer": self.tokens.profile(),
            "budget": {
                "beams": self.budget.beams(),
                "samples": self.budget.samples(),
                "suffix_len": self.budget.suffix_len(),
                "trials": self.budget.trials()
            }
        }

    @override
    def decode(self, suffix: list[int]) -> str:
        return self.tokens.decode(suffix)

    @override
    def score(self) -> float:
        if not math.isfinite(self.best):
            raise ValueError("BEAST has no scored suffix")
        return self.best

    @override
    def search(self, base: str, answer: str, journal: Journal) -> list[int] | None:
        """Extend each trial from its previous best, tracking the global minimum.

        Args:
            base (str): Exact Target request.
            answer (str): Prepared desired completion.
            journal (Journal): Independent attempt's events.

        Returns:
            list[int] | None: Best scored suffix, or no sampled candidate.

        Raises:
            ValueError: If scoring produces a nonfinite value.
        """
        self.budget.validate()
        continuation: list[int] = []
        best: list[int] | None = None
        for trial in range(
            1, self.budget.trials() + 1
        ):
            beams = self.sample(
                base, [continuation],
                self.budget.beams()
            )
            chosen: list[int] | None = None
            lowest = math.inf
            for depth in range(
                1, self.budget.suffix_len() + 1
            ):
                if not beams:
                    break
                scores = list(
                    zip(
                        self.tokens.loss(
                            [base + self.tokens.decode(beam) for beam in beams], answer
                        ),
                        beams, strict=True
                    )
                )
                if not all(math.isfinite(score) for score, _ in scores):
                    raise ValueError("BEAST search score must be finite")
                scores.sort(key=lambda item: item[0])
                beams = [item[1] for item in scores[:self.budget.beams()]]
                if scores[0][0] < lowest:
                    lowest, chosen = scores[0]
                if scores[0][0] < self.best:
                    self.best, best = scores[0]
                journal.record(
                    {
                        "event": "decision",
                        "action": "beast_beam",
                        "discarded": [],
                        "kept": len(beams),
                        "trial": trial,
                        "depth": depth,
                        "scores": [item[0] for item in scores],
                        "selected_score": scores[0][0]
                    }
                )
                if depth < self.budget.suffix_len():
                    beams = self.sample(
                        base, beams,
                        self.budget.samples()
                    )
            if chosen is None:
                break
            continuation = chosen
        return best
