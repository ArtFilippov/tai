"""Pinned refusal-phrase detector surface."""

from garak.attempt import Attempt

class MitigationBypass:
    def detect(self, attempt: Attempt) -> list[float | None]: ...
