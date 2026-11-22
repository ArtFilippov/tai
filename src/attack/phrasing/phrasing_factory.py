"""Deferred construction of one independent tense attack."""

from pathlib import Path
from typing import Protocol

from attack.attack import Attack
from attack.phrasing.phrasing_plan import PhrasingPlan


class PhrasingFactory(Protocol):
    """Bind an experiment plan to its reserved journal without sharing run state."""

    def __call__(self, plan: PhrasingPlan, path: Path) -> Attack:
        ...
