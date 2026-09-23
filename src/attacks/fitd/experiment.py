"""Compose FITD with shared execution and per-run observation."""

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, Unpack

from attacks.shared.assembly import validate_options
from attacks.shared.experiment import Experiment
from core.chats import LLMChat
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.messages import Message
from core.runs.journal import RunJournal
from core.runs.journal_ports import Journal

from .bootstrap import execution_settings
from .observation import observed_runtime
from .profile import FITDProfile


class ExperimentOptions(TypedDict, total=False):
    journal_dir: str | Path | None
    max_calls_per_conv: int
    journal_factory: Callable[..., Journal]


class FITDExperiment(Experiment):
    def __init__(
        self, target: LLMChat, attacker: LLMChat, base: Message,
        evaluator: Evaluator | EvaluatorSetup, **options: Unpack[ExperimentOptions],
    ) -> None:
        validate_options(options, frozenset({"journal_dir", "max_calls_per_conv", "journal_factory"}))
        setup = evaluator_setup(evaluator)
        rounds = options.get("max_calls_per_conv", 10)
        settings = execution_settings({"Target": target, "Attacker": attacker}, base, setup, {"max_calls_per_conv": rounds})
        super().__init__(
            settings, setup, journal_dir=options.get("journal_dir"), journal_factory=options.get("journal_factory", RunJournal),
            runtime_factory=lambda journal: observed_runtime(journal, FITDProfile(max_calls_per_conv=rounds)),
        )
