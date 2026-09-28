"""Observe GOAT judge decisions alongside shared run evidence."""

from collections.abc import Callable
from pathlib import Path
from typing import Unpack

from attacks.shared.assembly import validate_options
from attacks.shared.experiment import Experiment
from attacks.shared.observation import wrap_methods
from core.chats import LLMChat
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.messages import Message
from core.runs.journal import RunJournal
from core.runs.journal_ports import Journal

from .bootstrap import execution_settings
from .garak_adapter import GOATProbe, GOATRuntime
from .profile import PARAMETERS, GOATOptions, goat_profile


class ExperimentOptions(GOATOptions, total=False):
    journal_dir: str | Path | None
    journal_factory: Callable[..., Journal]


class GOATExperiment(Experiment):
    def __init__(self, target: LLMChat, attacker: LLMChat, base: Message, evaluator: Evaluator | EvaluatorSetup, **options: Unpack[ExperimentOptions]) -> None:
        validate_options(options, PARAMETERS | {"journal_dir", "journal_factory"})
        setup = evaluator_setup(evaluator)
        settings = execution_settings({"Target": target, "Attacker": attacker}, base, setup, options)
        profile = goat_profile(options)
        super().__init__(
            settings, setup, journal_dir=options.get("journal_dir"), journal_factory=options.get("journal_factory", RunJournal),
            runtime_factory=lambda journal: GOATRuntime(profile, wrap_methods(GOATProbe, journal, ("judge",))),
        )
