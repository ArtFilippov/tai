"""Journal AutoDAN fitness, native checks and result assessment."""

from collections.abc import Callable
from pathlib import Path
from typing import Unpack

from attacks.shared.assembly import validate_options
from attacks.shared.experiment import Experiment
from core.chats import LLMChat
from core.chats.scoring import require_scorer
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.messages import Message
from core.runs.journal import RunJournal
from core.runs.journal_ports import Journal

from .bootstrap import execution_settings
from .garak_adapter import AutoDANRuntime
from .profile import PARAMETERS, AutoDANOptions, autodan_profile


class ExperimentOptions(AutoDANOptions, total=False):
    journal_dir: str | Path | None
    journal_factory: Callable[..., Journal]


class AutoDANExperiment(Experiment):
    def __init__(self, target: LLMChat, attacker: LLMChat, base: Message,
                 evaluator: Evaluator | EvaluatorSetup, **options: Unpack[ExperimentOptions]) -> None:
        validate_options(options, PARAMETERS | {"journal_dir", "journal_factory"})
        setup = evaluator_setup(evaluator)
        settings = execution_settings({"Target": target, "Attacker": attacker}, base, setup, options)
        profile, scorer = autodan_profile(options), require_scorer(target)
        super().__init__(
            settings, setup, journal_dir=options.get("journal_dir"), journal_factory=options.get("journal_factory", RunJournal),
            runtime_factory=lambda journal: AutoDANRuntime(profile, scorer, target.model, lambda kind, data: journal.record(kind, **data)),
        )
