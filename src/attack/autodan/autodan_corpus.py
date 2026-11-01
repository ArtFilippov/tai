"""Source of bundled native AutoDAN prefix populations."""

from typing import Protocol


class AutoDANCorpus(Protocol):
    """Provide reference prefixes for one bounded native GA run."""

    def population(self, batch: int) -> list[str]:
        ...

    def description(self) -> str:
        ...
