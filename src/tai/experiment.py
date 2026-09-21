"""Experiment assembly and method wrappers; the strategy contains no logging.

Per-run subclasses intercept virtual dispatch before construction. Thus calls
from constructors and composite operations use the same wrappers as external
calls. No garak methods are monkeypatched and no private decision state is read.
"""

from collections.abc import Callable
from dataclasses import asdict
from functools import wraps
from pathlib import Path

from tai._diagnostics import DiagnosticRedactor
from tai._operations import WorkerOutcome
from tai.chats import LLMChat
from tai.evaluators import Evaluator, ModelEvaluator
from tai.fitd import FITDUser, _Execution
from tai.journal import RunJournal
from tai.messages import Message


def _wrap_methods[T](
    base: type[T], journal: RunJournal, names: tuple[str, ...]
) -> type[T]:
    def wrap(name):
        original = getattr(base, name)

        @wraps(original)
        def observed(self, *args, **kwargs):
            return journal.call(
                f"{base.__name__}.{name}",
                lambda: original(self, *args, **kwargs),
                project=(lambda value: value.decision)
                if name == "select_continuation"
                else (lambda value: value),
            )

        return observed

    return type(
        "Observed" + base.__name__, (base,), {name: wrap(name) for name in names}
    )


def _execution_type(journal: RunJournal, directory: Path) -> type[_Execution]:
    class ObservedExecution(_Execution):
        def _admit(self, started):
            super()._admit(started)
            journal.bind(started.run_id)

        def _initialize(self, run_id):
            journal.begin(run_id, directory / run_id)
            super()._initialize(run_id)
            journal.configure()

        def _ensure_open(self):
            # A library may catch an exception. Its next working boundary must
            # still fail with the original journal error, before any work.
            journal.check()
            return super()._ensure_open()

        def _request_stop(self):
            stopped = super()._request_stop()
            if stopped and journal.error is None:
                journal.enqueue("stop_requested")
            return stopped

        def _exchange(self, context, call_id, message, result_id):
            operation = super()._exchange
            return journal.call(
                "exchange",
                lambda: operation(context, call_id, message, result_id),
                context=context,
                call_id=call_id,
                message=message,
                result_id=result_id,
            )

        def _assess(self, result, context, checkpoint_id):
            operation = super()._assess
            return journal.call(
                "assessment",
                lambda: operation(result, context, checkpoint_id),
                context=context,
                result_id=result.id,
                checkpoint_id=checkpoint_id,
            )

        def _run_strategy(self, **kwargs):
            from tai._fitd import _FITD, _Target

            target_type = _wrap_methods(
                _Target, journal, ("save", "exchange", "synchronize")
            )

            class ObservedProbe(_FITD):
                def raise_if_interrupted(self):
                    journal.check()
                    return super().raise_if_interrupted()

            probe_type = _wrap_methods(
                ObservedProbe,
                journal,
                ("check_refusal", "check_alignment", "select_continuation"),
            )
            return super()._run_strategy(target_type=target_type, probe_type=probe_type)

        def _finalize_status(self, status, result, revision):
            super()._finalize_status(status, result, revision)
            journal.save_status(status, result, revision)

        def _complete(self, outcome: WorkerOutcome):
            super()._complete(outcome)
            journal.finish(outcome)

    return _wrap_methods(
        ObservedExecution, journal, ("_checkpoint", "_restore", "_reset", "_publish")
    )


class FITDExperiment(FITDUser):
    """Assemble a FITD User with a fresh, independently readable journal per run.

    Lifecycle and result methods are the User interface. ``journal`` selects the
    current run; retained journal references remain valid across repeated starts.
    ``journals`` contains the admitted runs, in order.
    """

    def __init__(
        self,
        target: LLMChat,
        attacker: LLMChat,
        base: Message,
        evaluator: Evaluator,
        *,
        artifacts_dir: str | Path = "runs",
        max_calls_per_conv: int = 10,
        journal_factory: Callable[..., RunJournal] = RunJournal,
    ) -> None:
        self._journals: dict[_Execution, RunJournal] = {}
        self._journal_factory = journal_factory
        super().__init__(
            target,
            attacker,
            base,
            evaluator,
            artifacts_dir=artifacts_dir,
            max_calls_per_conv=max_calls_per_conv,
        )

    @property
    def journal(self) -> RunJournal:
        with self._lock:
            return self._journals[self._execution]

    @property
    def journals(self) -> tuple[RunJournal, ...]:
        with self._lock:
            return tuple(
                journal
                for journal in self._journals.values()
                if journal.status().run_id is not None
            )

    def _new_execution(
        self, execution_type: type[_Execution] | None = None
    ) -> _Execution:
        chats = {"Target": self._target, "Attacker": self._attacker}
        if isinstance(self._evaluator, ModelEvaluator):
            chats["Evaluator"] = self._evaluator.chat
        redactor = DiagnosticRedactor.combine(
            *(chat._redactor for chat in chats.values())
        )

        def participants():
            return {
                role: chat._diagnostic_settings(redactor)
                for role, chat in chats.items()
            }

        def settings():
            from tai._fitd import GARAK_REVISION

            return {
                "garak_revision": GARAK_REVISION,
                **asdict(self._profile),
                "evaluator": self._evaluator.settings(),
                "base": self._base,
                **participants(),
            }

        journal = self._journal_factory(
            redactor=redactor,
            max_calls_per_conv=self._profile.max_calls_per_conv,
            settings=settings,
            final_settings=participants,
        )
        execution = super()._new_execution(
            _execution_type(journal, self._artifacts_dir)
        )
        self._journals[execution] = journal
        return execution
