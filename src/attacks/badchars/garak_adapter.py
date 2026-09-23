"""A generator-only facade over the pinned native BadCharacters deletion path."""

from collections.abc import Iterator
from dataclasses import dataclass

from garak.probes.badchars import BadCharacters

from attacks.shared.contracts import Candidate
from core.primitives.messages import Message
from core.runs.operations import Cause

from .profile import DeletionProfile


@dataclass
class _DeletionGenerator(BadCharacters):
    # Initialize only the inputs of the pure native generator. BadCharacters'
    # constructor loads a bundled corpus, generates all categories, and samples.
    # None of that belongs to generating deletion variants of one explicit base.
    perturbation_budget: int
    max_position_candidates: int
    max_ascii_variants: int

    def candidates(self, base: Message) -> Iterator[Candidate]:
        for text, details in self._generate_deletion_variants(base.to_string()):
            request = base.spawn(text)
            metadata: dict[str, object] = {"bad_character_category": "deletion", **details}
            yield Candidate(request, Cause("badchars_deletion_variant", request.id, metadata=metadata))


def candidates(base: Message, *, profile: DeletionProfile) -> Iterator[Candidate]:
    native = _DeletionGenerator(profile.perturbation_budget, profile.max_position_candidates, profile.max_ascii_variants)
    return native.candidates(base)
