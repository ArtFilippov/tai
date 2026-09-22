"""Experiment assembly and method wrappers; the strategy contains no logging.

Per-run subclasses intercept virtual dispatch before construction. Thus calls
from constructors and composite operations use the same wrappers as external
calls. No garak methods are monkeypatched and no private decision state is read.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TypedDict, Unpack, override

from core.chats import LLMChat
from core.evaluation import Evaluator
from core.evaluation.configuration import EvaluatorSetup, evaluator_setup
from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.messages import Message
from core.runs.journal import RunJournal
from core.runs.journal_ports import Journal

from .bootstrap import execution_settings, validate_options
from .execution import Execution
from .execution_state import ExecutionSettings
from .observation import observed_execution, observed_runtime
from .user import FITDUser


class ExperimentOptions(TypedDict, total=False):
    journal_dir: str | Path | None
    max_calls_per_conv: int
    journal_factory: Callable[..., Journal]


@dataclass
class _Journals:
    setup: EvaluatorSetup
    directory: Path | None
    factory: Callable[..., Journal]
    runs: dict[Execution, Journal] = field(default_factory=dict[Execution, Journal])


class FITDExperiment(FITDUser):
    """Assemble a FITD User with a fresh, independently readable journal per run.

    Lifecycle and result methods are the User interface. ``journal`` selects the
    current run; retained journal references remain valid across repeated starts.
    ``journals`` contains the admitted runs, in order.
    """

    def __init__(
        self, target: LLMChat, attacker: LLMChat, base: Message,
        evaluator: Evaluator | EvaluatorSetup, **options: Unpack[ExperimentOptions],
    ) -> None:
        """Prepare independently observed executions with the pinned runtime.

        Args:
            target (LLMChat): The independent Target conversation.
            attacker (LLMChat): The independent Attacker conversation.
            base (Message): Explicit task whose provenance is retained.
            evaluator (Evaluator | EvaluatorSetup): Result assessment and assembly metadata.
            **options (Unpack[ExperimentOptions]): Journal directory, factory, and round budget.
        """
        validate_options(options, frozenset({"journal_dir", "max_calls_per_conv", "journal_factory"}))
        directory = options.get("journal_dir")
        setup = evaluator_setup(evaluator)
        self._observation = _Journals(
            setup, Path(directory) if directory is not None else None, options.get("journal_factory", RunJournal),
        )
        super().__init__(execution_settings(
            {"Target": target, "Attacker": attacker}, base, setup,
            {"max_calls_per_conv": options.get("max_calls_per_conv", 10)},
        ))

    @property
    def journal(self) -> Journal:
        with self._lock:
            return self._observation.runs[self._execution]

    @property
    def journals(self) -> tuple[Journal, ...]:
        with self._lock:
            return tuple(journal for journal in self._observation.runs.values() if journal.status().run_id is not None)

    @override
    def new_execution(self, execution_type: Callable[[ExecutionSettings], Execution] = Execution) -> Execution:
        journal = self._new_journal()
        factory = observed_execution(journal, self._observation.directory)
        execution = factory(replace(self._settings, runtime=observed_runtime(journal)))
        self._observation.runs[execution] = journal
        return execution

    def _new_journal(self) -> Journal:
        chats = self._settings.chats
        setup = self._observation.setup
        redactor = DiagnosticRedactor.combine(*(chat.redactor for chat in chats.values()))

        def participants() -> dict[str, object]:
            return {role: chat.diagnostic_settings(redactor) for role, chat in chats.items()}

        def settings() -> dict[str, object]:
            return {
                "garak_revision": self._settings.runtime.revision, **asdict(self._settings.profile),
                "evaluator": setup.settings(), "base": self.base, **participants(),
            }

        return self._observation.factory(
            redactor=redactor, max_calls_per_conv=self._settings.profile.max_calls_per_conv,
            settings=settings, final_settings=participants,
        )
