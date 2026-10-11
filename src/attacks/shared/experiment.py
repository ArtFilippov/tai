"""Experiment assembly and method wrappers; the strategy contains no logging.

Per-run subclasses intercept virtual dispatch before construction. Thus calls
from constructors and composite operations use the same wrappers as external
calls. No garak methods are monkeypatched and no private decision state is read.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypedDict, Unpack, override

from core.evaluation.configuration import EvaluatorSetup
from core.runs.journal import RunJournal
from core.runs.journal_ports import Journal

from .assembly import validate_options
from .contracts import StrategyRuntime
from .execution import Execution
from .execution_state import ExecutionSettings
from .observation import observed_execution
from .user import ExecutionUser


class ObservationOptions(TypedDict, total=False):
    journal_dir: str | Path | None
    journal_factory: Callable[..., Journal]
    runtime_factory: Callable[[Journal], StrategyRuntime]


@dataclass
class _Journals:
    setup: EvaluatorSetup
    directory: Path | None
    factory: Callable[..., Journal]
    runtime_factory: Callable[[Journal], StrategyRuntime]
    runs: dict[Execution, Journal] = field(default_factory=dict[Execution, Journal])


class Experiment(ExecutionUser):
    """Assemble a User with a fresh, independently readable journal per run.

    Lifecycle and result methods are the User interface. ``journal`` selects the
    current run; retained journal references remain valid across repeated starts.
    ``journals`` contains the admitted runs, in order.
    """

    def __init__(
        self, settings: ExecutionSettings, setup: EvaluatorSetup, **options: Unpack[ObservationOptions],
    ) -> None:
        """Prepare independently observed executions with the pinned runtime.

        Args:
            settings (ExecutionSettings): Participants, base, runtime and parameters.
            setup (EvaluatorSetup): Result assessment and assembly metadata.
            **options (Unpack[ObservationOptions]): Journal directory and observation factories.
        """
        validate_options(options, frozenset({"journal_dir", "journal_factory", "runtime_factory"}))
        directory = options.get("journal_dir")
        self._observation = _Journals(
            setup, Path(directory) if directory is not None else None, options.get("journal_factory", RunJournal),
            options.get("runtime_factory", lambda journal: settings.runtime),
        )
        super().__init__(settings)

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
        execution = factory(replace(self._settings, runtime=self._observation.runtime_factory(journal)))
        self._observation.runs[execution] = journal
        return execution

    def _new_journal(self) -> Journal:
        chats = self._settings.chats
        configuration = self._settings.configuration
        setup = self._observation.setup
        redactor = chats["Target"].redactor.combine(*(chat.redactor for chat in chats.values()))

        def participants() -> dict[str, object]:
            return {role: chat.diagnostic_settings(redactor) for role, chat in chats.items()}

        def settings() -> dict[str, object]:
            return {
                "garak_revision": self._settings.runtime.revision, **configuration.settings(),
                "evaluator": setup.settings(), "base": self.base, **participants(),
            }

        return self._observation.factory(
            redactor=redactor, max_calls_per_conv=self._settings.configuration.max_calls_per_conv,
            settings=settings, final_settings=participants,
        )
