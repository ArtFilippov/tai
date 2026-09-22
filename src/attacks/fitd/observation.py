"""Typed operation observation assembled outside the FITD execution layer."""

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import cast, override

from core.primitives.messages import Message
from core.evaluation.observation import observe_evaluation
from core.runs import UserState
from core.runs.journal_ports import Journal
from core.runs.operations import Context, EvaluationResult, Exchange, RunStarted, WorkerOutcome

from .contracts import Continuation
from .execution import Execution
from .execution_state import ExecutionSettings
from .garak_adapter import FITDProbe, GarakRuntime
from .target import FITDTarget


def _project(value: object) -> object:
    return value.decision if isinstance(value, Continuation) else value


def _wrap_method(base: type, name: str, journal: Journal) -> Callable[..., object]:
    original = cast(Callable[..., object], getattr(base, name))

    @wraps(original)
    def observed(receiver: object, *args: object, **kwargs: object) -> object:
        return journal.call(
            f"{base.__name__}.{name}", lambda: original(receiver, *args, **kwargs), project=_project,
        )

    return observed


def wrap_methods[T](
    base: type[T], journal: Journal, names: tuple[str, ...],
    overrides: dict[str, Callable[..., object]] | None = None,
) -> type[T]:
    methods = {name: _wrap_method(base, name, journal) for name in names}
    return cast(type[T], type("Observed" + base.__name__, (base,), methods | (overrides or {})))


class ObservedExecution(Execution):
    def __init__(self, settings: ExecutionSettings, journal: Journal, directory: Path | None) -> None:
        self._journal, self._directory = journal, directory
        super().__init__(settings)

    @override
    def admit(self, started: RunStarted) -> None:
        super().admit(started)
        self._journal.bind(started.run_id)

    @override
    def initialize(self, run_id: str) -> None:
        self._journal.begin(run_id, self._directory / run_id if self._directory is not None else None)
        super().initialize(run_id)
        self._journal.configure()

    @override
    def ensure_open(self) -> None:
        self._journal.check()
        super().ensure_open()

    @override
    def request_stop(self) -> bool:
        stopped = super().request_stop()
        if stopped and self._journal.error is None:
            self._journal.enqueue("stop_requested")
        return stopped

    @override
    def exchange(self, context: Context, call_id: str, message: Message) -> Exchange:
        operation = super().exchange
        return self._journal.call(
            "exchange", lambda: operation(context, call_id, message),
            context=context, call_id=call_id, message=message,
        )

    @override
    def assess(self, result: Message, context: Context, checkpoint_id: str) -> EvaluationResult:
        operation = super().assess
        with observe_evaluation(self._journal, context, self.base, result) as evaluation_id:
            return self._journal.call(
                "assessment", lambda: operation(result, context, checkpoint_id),
                context=context, base=self.base, result=result, evaluation_id=evaluation_id,
                result_id=result.id, checkpoint_id=checkpoint_id,
            )

    @override
    def finalize(self, state: UserState, reason: str) -> None:
        self._journal.capture_final_settings()
        super().finalize(state, reason)

    @override
    def complete(self, outcome: WorkerOutcome) -> None:
        super().complete(outcome)
        _ = self._journal.finish(outcome)


def observed_execution(journal: Journal, directory: Path | None) -> Callable[[ExecutionSettings], Execution]:
    observed_type = wrap_methods(ObservedExecution, journal, ("checkpoint", "restore", "reset", "publish"))
    return lambda settings: observed_type(settings, journal, directory)


def observed_runtime(journal: Journal) -> GarakRuntime:
    def check_probe(probe: FITDProbe) -> None:
        journal.check()
        FITDProbe.raise_if_interrupted(probe)

    return GarakRuntime(
        wrap_methods(FITDTarget, journal, ("save", "exchange", "synchronize")),
        wrap_methods(
            FITDProbe, journal, ("check_refusal", "check_alignment", "select_continuation"),
            {"raise_if_interrupted": check_probe},
        ),
    )
