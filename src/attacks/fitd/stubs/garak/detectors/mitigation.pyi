"""The built-in refusal-prefix detector used by native FITD."""

from typing import override
from ..attempt import Attempt
from .base import StringDetector

class Prefixes(StringDetector):
    def __init__(self, config_root: object = ...) -> None: ...
    @override
    def detect(self, attempt: Attempt) -> list[float | None]: ...
