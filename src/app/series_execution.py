"""Sequential ALERT execution with injected construction, storage, and stop ports."""

from dataclasses import dataclass, field

from alert.accounting import COMPLETED, SeriesReport, build_series_report, classify_outcome
from alert.records import AlertRecord
from core.primitives.messages import Message
from core.runs import RunSnapshot

from .contracts import RunHandle
from .failure_capture import Failure
from .lifecycle import join_admitted, wait_for_run
from .series_contracts import SeriesDependencies, SeriesResult, snapshot_outcome


@dataclass
class Attempt:
    base: Message
    started_at: str
    start_time: float
    handle: RunHandle | None = None
    phase: str = "create_experiment"
    error: dict[str, str] | None = None

    def run(self, dependencies: SeriesDependencies) -> bool:
        self.handle = dependencies.create(self.base)
        if dependencies.stop.interrupted:
            return False
        self.phase = "start"
        self.handle.start()
        self.phase = "wait"
        wait_for_run(self.handle, dependencies.stop)
        return True

    def join(self) -> None:
        if self.handle is not None:
            join_admitted(self.handle)


def _failure(error: Exception, phase: str, dependencies: SeriesDependencies) -> dict[str, str]:
    return {"phase": phase, "type": type(error).__name__, "message": dependencies.redact(str(error))}


def _capture(attempt: Attempt, dependencies: SeriesDependencies) -> dict[str, object]:
    snapshot, storage_error = None, None
    if (handle := attempt.handle) is not None:
        status = handle.status()
        snapshot = RunSnapshot(status, None) if status.run_id is None else handle.snapshot()
        if status.run_id is not None:
            with Failure() as failure:
                handle.save()
            if failure.error is not None:
                storage_error = _failure(failure.error, "save", dependencies)
        if attempt.error is None and (error := handle.journal().error) is not None:
            attempt.error = _failure(error, "journal", dependencies)
    outcome = snapshot_outcome(snapshot)
    reason = outcome["reason"]
    outcome.update(dependencies.store.inspect(snapshot), error=attempt.error, storage_error=storage_error,
                   reason=dependencies.redact(reason) if isinstance(reason, str) else None)
    return outcome


def _attempt(record: AlertRecord, dependencies: SeriesDependencies) -> dict[str, object] | None:
    attempt = Attempt(Message(record.base_request), dependencies.clock.now(), dependencies.clock.elapsed())
    try:
        return _attempt_outcome(attempt, dependencies)
    finally:
        attempt.join()


def _attempt_outcome(attempt: Attempt, dependencies: SeriesDependencies) -> dict[str, object] | None:
    with Failure() as failure:
        if not attempt.run(dependencies):
            return None
    if failure.error is not None:
        attempt.error = _failure(failure.error, attempt.phase, dependencies)
        attempt.join()
    outcome = _capture(attempt, dependencies)
    return {
        "base_id": attempt.base.id, "started_at": attempt.started_at,
        "finished_at": dependencies.clock.now(),
        "duration_seconds": dependencies.clock.elapsed() - attempt.start_time,
        **outcome,
    }


@dataclass
class SeriesExecution:
    records: tuple[AlertRecord, ...]
    dependencies: SeriesDependencies
    index: dict[str, object] = field(default_factory=dict[str, object])
    entries: list[dict[str, object]] = field(default_factory=list[dict[str, object]])

    def _save_snapshot(self) -> SeriesReport:
        self.index["interrupted"] = self.dependencies.stop.interrupted
        report = build_series_report(self.index)
        self.index["complete"] = report["complete"]
        self.dependencies.store.persist(self.index, report)
        return report

    def save(self) -> SeriesReport:
        # Only one false -> true transition is possible. A signal in either atomic
        # write invalidates that snapshot, including at the final-record boundary.
        report = self._save_snapshot()
        if self.index["interrupted"] != self.dependencies.stop.interrupted:
            report = self._save_snapshot()
        return report

    def _run_records(self, planned: list[dict[str, object]]) -> None:
        for sequence, record in enumerate(self.records, 1):
            if self.dependencies.stop.interrupted:
                break
            if (outcome := _attempt(record, self.dependencies)) is None:
                break
            entry = {**planned[sequence - 1], "sequence": sequence, **outcome,
                     "classification": classify_outcome(outcome)}
            self.entries.append(entry)
            _ = self.save()

    def run(self) -> SeriesResult:
        self.index = self.dependencies.store.initialize()
        began = self.dependencies.clock.elapsed()
        planned: list[dict[str, object]] = [
            {"id": record.id, "source_category": record.source_category,
             "micro_category": record.micro_category, "macro_category": record.macro_category}
            for record in self.records
        ]
        self.index.update(started_at=self.dependencies.clock.now(), finished_at=None, duration_seconds=None,
                          interrupted=False, complete=False, planned=planned, entries=self.entries)
        _ = self.save()  # Validate shared storage before the first start.
        self._run_records(planned)
        self.dependencies.store.finish()
        self.index.update(finished_at=self.dependencies.clock.now(), duration_seconds=self.dependencies.clock.elapsed() - began)
        report = self.save()
        # Seal only after final files exist. A last signal before sealing still
        # needs persistence, rather than exit 130 with a stale complete index.
        self.dependencies.stop.seal()
        if self.index["interrupted"] != self.dependencies.stop.interrupted:
            report = self.save()
        failed = any(entry["error"] or entry["storage_error"] or entry["artifact_errors"]
                     or entry["state"] != COMPLETED for entry in self.entries)
        return SeriesResult(self.dependencies.store.directory, report, 130 if self.dependencies.stop.interrupted else int(failed))
