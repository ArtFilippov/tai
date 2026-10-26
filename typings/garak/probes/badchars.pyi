"""Pinned deletion generation surface of garak's BadCharacters."""

from collections.abc import Iterator
from typing import TypedDict


class DeletionDetails(TypedDict):
    count: int
    operations: dict[str, list[int]]


class BadCharacters:
    perturbation_budget: int
    max_position_candidates: int
    max_ascii_variants: int

    def _generate_deletion_variants(self, payload: str) -> Iterator[tuple[str, DeletionDetails]]: ...
