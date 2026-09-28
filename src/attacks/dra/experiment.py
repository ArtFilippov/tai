"""DRA observation with the shared journal and lifecycle."""

from collections.abc import Callable
from pathlib import Path
from typing import Unpack

from attacks.shared.assembly import validate_options
from attacks.shared.experiment import Experiment
from core.chats import LLMChat
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.messages import Message
from core.runs.journal import RunJournal
from core.runs.journal_ports import Journal

from .bootstrap import execution_settings
from .profile import PARAMETERS, DRAOptions


class ExperimentOptions(DRAOptions, total=False):
    journal_dir: str | Path | None
    journal_factory: Callable[..., Journal]


class DRAExperiment(Experiment):
    def __init__(self, target: LLMChat, base: Message, evaluator: Evaluator | EvaluatorSetup, **options: Unpack[ExperimentOptions]) -> None:
        validate_options(options, PARAMETERS | {"journal_dir", "journal_factory"})
        setup = evaluator_setup(evaluator)
        journal_factory = options.get("journal_factory", RunJournal)
        super().__init__(
            execution_settings(target, base, setup, options), setup,
            journal_dir=options.get("journal_dir"), journal_factory=journal_factory,
        )
