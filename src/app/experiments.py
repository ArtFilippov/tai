"""Bind composed experiments to the execution and external persistence ports."""

from functools import partial
from pathlib import Path

from attacks.fitd import FITDExperiment
from core.runs.storage import capture_run, save_run

from .contracts import RunHandle


def _persist(user: FITDExperiment, directory: Path) -> None:
    _ = save_run(capture_run(user, journal=user.journal), directory)


def bind_experiment(user: FITDExperiment, directory: Path) -> RunHandle:
    return RunHandle(user, lambda: user.journal, partial(_persist, user, directory))
