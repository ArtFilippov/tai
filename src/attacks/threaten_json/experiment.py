"""Attach per-run JSON format comparison to the standard run journal."""

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
from .threaten_json_runtime import ThreatenJSONRuntime


class ExperimentOptions(TypedDict, total=False):
    journal_dir: str | Path | None
    journal_factory: Callable[..., Journal]


class ThreatenJSONExperiment(Experiment):
    """Keep control evidence and format comparison in a fresh journal per run."""

    def __init__(
        self, target: LLMChat, base: Message, evaluator: Evaluator | EvaluatorSetup,
        **options: Unpack[ExperimentOptions],
    ) -> None:
        """Prepare a reusable, observed User.

        Args:
            target (LLMChat): Target conversation.
            base (Message): Explicit JSON-answerable request.
            evaluator (Evaluator | EvaluatorSetup): Substantive answer judge.
            **options (Unpack[ExperimentOptions]): Journal destination and factory.
        """
        validate_options(options, frozenset({"journal_dir", "journal_factory"}))
        setup = evaluator_setup(evaluator)
        super().__init__(
            execution_settings(target, base, setup), setup,
            journal_dir=options.get("journal_dir"), journal_factory=options.get("journal_factory", RunJournal),
            runtime_factory=lambda journal: ThreatenJSONRuntime(
                lambda control, result, before, after: journal.record(
                    "json_comparison", control_id=control.id, result_id=result.id,
                    control_parseable=before, result_parseable=after,
                )
            ),
        )
