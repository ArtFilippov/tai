"""Actual ancestry of the native string detector."""

from collections.abc import Iterable
from ..attempt import Attempt
from ..configurable import Configurable

class Detector(Configurable):
    def __init__(self, config_root: object = ...) -> None: ...
    def detect(self, attempt: Attempt) -> Iterable[float | None]: ...

class StringDetector(Detector):
    def __init__(self, substrings: list[str], config_root: object = ...) -> None: ...
