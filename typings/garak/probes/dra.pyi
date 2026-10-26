"""Pinned DRA tool list and native puzzle methods used by tai."""

import random

TOOL_LIST: list[str]


class DRA:
    letter_map: dict[str, list[str]]
    moderation_cache: dict[str, bool]
    toxic_trunc: float
    benign_trunc: float
    benign_full_token_prob: float

    def _prefix_gen(self, question: str) -> tuple[str, str]: ...
    def _suffix_gen(self, question: str, template: str) -> str: ...
