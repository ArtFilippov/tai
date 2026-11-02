"""Hard bounds and reproducible traversal parameters."""

from dataclasses import dataclass
from typing import override

from attack.wordnet.wordnet_limits import WordnetLimits


@dataclass
class WordnetBudget(WordnetLimits):
    """Limit visited synsets and Target exchanges in one root attempt.

    Attributes:
        node_limit (int): Maximum visited synsets.
        call_limit (int): Maximum Target exchanges.
        traversal_seed (int): Reproducible traversal seed.
    """

    node_limit: int
    call_limit: int
    traversal_seed: int

    @override
    def nodes(self) -> int:
        return self.node_limit

    @override
    def calls(self) -> int:
        return self.call_limit

    @override
    def seed(self) -> int:
        return self.traversal_seed

    @override
    def validate(self) -> None:
        """Reject limits that cannot bound a search.

        Raises:
            ValueError: If either bound is not positive.
        """
        if self.node_limit < 1 or self.call_limit < 1:
            raise ValueError("WordNet node and Target-call limits must be positive")
