"""Adapter to the pinned garak Bad Characters deletion generator."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from garak.probes.badchars import BadCharacters

@dataclass(eq=False)
class NativeDeletion(BadCharacters):
    """Use native deletion generation without loading the garak corpus.

    Attributes:
        perturbation_budget (int): Maximum inserted sequences.
        max_position_candidates (int): Maximum selected insertion sites.
        max_ascii_variants (int): Maximum ASCII substitutions per site.
    """

    perturbation_budget: int
    max_position_candidates: int
    max_ascii_variants: int

    def generate(self, base: str) -> Iterator[Mapping[str, object]]:
        """Yield native deletion text with its operation metadata.

        Args:
            base (str): Original request to perturb.

        Yields:
            Mapping[str, object]: Request and native transformation details.
        """
        for text, details in self._generate_deletion_variants(base):
            yield {
                "text": text,
                "count": details["count"],
                "operations": details["operations"]
            }
