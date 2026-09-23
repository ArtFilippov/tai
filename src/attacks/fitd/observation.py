"""Observation of native FITD decisions, separate from shared run observation."""

from attacks.shared.observation import wrap_methods
from core.runs.journal_ports import Journal

from .contracts import Continuation
from .garak_adapter import FITDProbe, GarakRuntime
from .profile import FITDProfile
from .target import FITDTarget


def _project(value: object) -> object:
    return value.decision if isinstance(value, Continuation) else value


def observed_runtime(journal: Journal, profile: FITDProfile) -> GarakRuntime:
    def check_probe(probe: FITDProbe) -> None:
        journal.check()
        FITDProbe.raise_if_interrupted(probe)

    return GarakRuntime(
        wrap_methods(FITDTarget, journal, ("save", "exchange", "synchronize")),
        wrap_methods(
            FITDProbe, journal, ("check_refusal", "check_alignment", "select_continuation"),
            {"raise_if_interrupted": check_probe}, project=_project,
        ),
        profile,
    )
