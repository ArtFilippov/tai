"""Pure configuration providers and constructor options for a run journal."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypedDict


class JournalOptions(TypedDict, total=False):
    max_calls_per_conv: int | None
    settings: Callable[[], Mapping[str, object]]
    final_settings: Callable[[], Mapping[str, object]]


@dataclass(frozen=True)
class JournalConfiguration:
    source: Callable[[], Mapping[str, object]]
    final_source: Callable[[], Mapping[str, object]]
