"""Only the native deletion generator surface; inheritance remains real."""

from collections.abc import Iterator
from typing import TypedDict

from .base import Probe

class _DeletionOperations(TypedDict):
    positions: list[int]
    ascii_codes: list[int]

class _DeletionDetails(TypedDict):
    count: int
    operations: _DeletionOperations

class BadCharacters(Probe):
    perturbation_budget: int
    max_position_candidates: int
    max_ascii_variants: int
    def __init__(self, config_root: object = ...) -> None: ...
    def _generate_deletion_variants(self, payload: str) -> Iterator[tuple[str, _DeletionDetails]]: ...
