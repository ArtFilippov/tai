"""Compose real role exchanges and a neutral pair through the evaluator contract."""

from collections.abc import Callable
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import monotonic

from core.evaluation.observation import observe_evaluation
from core.runs.journal import RunJournal

from .contracts import Clock, ReadinessServices
from .profiles import ParticipantProfile
from .readiness_session import ReadinessSession
from .runtime import utc_now


def run_preflight(
    profile: ParticipantProfile, report: Path | None = None, *,
    journal_factory: Callable[..., RunJournal] = RunJournal,
) -> int:
    journal = journal_factory(redactor=profile.redactor())
    services = ReadinessServices(journal, partial(observe_evaluation, journal), Clock(utc_now, monotonic), version("tai"))
    data, rendered = ReadinessSession(profile, services).run(report)
    print(rendered, end="")
    return 0 if data["ready"] else 1
