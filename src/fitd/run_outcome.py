"""Recovered run facts, without an inferred or reassessed result."""

from typing import TypedDict


class RunOutcome(TypedDict):
    """The recorded outcome and per-role call counts of one run.

    Attributes:
        status: Recorded completion state.
        text: Exactly the presented text, or none when absent.
        assessment: Recorded verdict, or none when absent.
        calls: Number of observed calls by role.
    """

    status: str
    text: str | None
    assessment: bool | None
    calls: dict[str, int]
