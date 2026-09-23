"""Compose Adaptive Attacks with the existing lifecycle and run journal."""

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, Unpack

from attacks.shared.assembly import validate_options
from attacks.shared.experiment import Experiment
from core.chats import LLMChat
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.messages import Message

from core.runs.journal_ports import Journal
from .bootstrap import execution_settings


class ExperimentOptions(TypedDict, total=False):
    journal_dir: str | Path | None
    journal_factory: Callable[..., Journal]


class AdaptiveAttacksExperiment(Experiment):
    def __init__(
        self, target: LLMChat, base: Message, evaluator: Evaluator | EvaluatorSetup, **options: Unpack[ExperimentOptions],
    ) -> None:
        validate_options(options, frozenset({"journal_dir", "journal_factory"}))
        setup = evaluator_setup(evaluator)
        super().__init__(execution_settings(target, base, setup), setup, **options)
