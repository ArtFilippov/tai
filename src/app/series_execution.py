"""Single-writer coordinator of bounded, independently owned FITD workers."""

from dataclasses import dataclass, field
from itertools import count
from threading import Event

from alert.accounting import COMPLETED, Classification, SeriesReport, build_series_report, classify_outcome
from alert.records import AlertRecord
from core.primitives.messages import Message

from .failure_capture import Failure
from .lifecycle import POLL_SECONDS
from .series_attempt import Attempt
from .series_contracts import SeriesDependencies, SeriesIdentity, SeriesResult


def validate_jobs(jobs: object) -> None:
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")


def _display_state(outcome: dict[str, object], classification: Classification) -> str:
    if outcome["error"] or outcome["storage_error"] or outcome["artifact_errors"]:
        return "ERROR"
    if outcome["state"] == COMPLETED:
        return {True: "SUCCESS", False: "UNSUCCESSFUL", None: "UNEVALUATED"}[classification["success"]]
    return str(outcome["state"])


@dataclass
class SeriesExecution:
    records: tuple[AlertRecord, ...]
    dependencies: SeriesDependencies
    jobs: int = 4
    index: dict[str, object] = field(default_factory=dict[str, object])
    entries: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    active: dict[int, Attempt] = field(default_factory=dict[int, Attempt])
    stopping: bool = False

    def __post_init__(self) -> None:
        validate_jobs(self.jobs)

    def _save_snapshot(self) -> SeriesReport:
        self.index["interrupted"] = self.dependencies.stop.interrupted
        report = build_series_report(self.index)
        self.index["complete"] = report["complete"]
        self.dependencies.store.persist(self.index, report)
        return report

    def save(self) -> SeriesReport:
        report = self._save_snapshot()
        if self.index["interrupted"] != self.dependencies.stop.interrupted:
            report = self._save_snapshot()
        return report

    def _admit(self) -> None:
        sequence = len(self.entries) + 1
        record = self.records[sequence - 1]
        attempt = Attempt(SeriesIdentity(sequence, record.micro_category), Message(record.base_request), self.dependencies.clock.elapsed())
        started = self.dependencies.clock.now()
        if not attempt.admit(self.dependencies):
            return
        self.active[sequence] = attempt
        self.entries.append({
            "id": record.id, "source_category": record.source_category,
            "micro_category": record.micro_category, "macro_category": record.macro_category,
            "sequence": sequence, "base_id": attempt.base.id, "started_at": started, **attempt.active(),
        })
        _ = self.save()

    def _request_stops(self) -> None:
        # Request every stop before waiting for any worker or doing console I/O.
        for attempt in self.active.values():
            attempt.stop(self.dependencies)

    def _stop_all(self) -> None:
        self._request_stops()
        self.stopping = True
        for sequence in self.active:
            self.dependencies.display(sequence, "STOPPING")
        _ = self.save()

    def _collect(self) -> None:
        for sequence, attempt in tuple(self.active.items()):
            if attempt.done():
                self._capture(sequence, attempt)
                _ = self.save()

    def _capture(self, sequence: int, attempt: Attempt) -> None:
        self.dependencies.display(sequence, "SAVING")
        outcome = attempt.capture(self.dependencies)
        classification = classify_outcome(outcome)
        self.entries[sequence - 1].update(outcome, classification=classification)
        del self.active[sequence]
        self.dependencies.display(sequence, _display_state(outcome, classification))

    def _run_records(self) -> None:
        for _ in count():
            if self.dependencies.stop.interrupted and not self.stopping:
                self._stop_all()
            self._collect()
            if self._advance():
                return

    def _advance(self) -> bool:
        if self._can_admit():
            self._admit()
        elif not self.active:
            return True
        else:
            _ = Event().wait(POLL_SECONDS)
        return False

    def _can_admit(self) -> bool:
        return (not self.dependencies.stop.interrupted and len(self.active) < self.jobs
                and len(self.entries) < len(self.records))

    def _drain(self) -> None:
        self._request_stops()
        for sequence, attempt in tuple(self.active.items()):
            attempt.join()
            # Shared storage/display failure must not skip any user's final save.
            with Failure():
                outcome = attempt.capture(self.dependencies)
                self.entries[sequence - 1].update(outcome, classification=classify_outcome(outcome))
            del self.active[sequence]

    def _initialize(self) -> None:
        self.index = self.dependencies.store.initialize()
        planned: list[dict[str, object]] = [
            {"id": record.id, "source_category": record.source_category,
             "micro_category": record.micro_category, "macro_category": record.macro_category}
            for record in self.records
        ]
        self.index.update(schema_version=2, started_at=self.dependencies.clock.now(), finished_at=None,
                          duration_seconds=None, interrupted=False, complete=False, planned=planned,
                          entries=self.entries, jobs=self.jobs, effective_jobs=min(self.jobs, len(self.records)))
        _ = self.save()

    def _finish(self, began: float) -> SeriesResult:
        self.dependencies.store.finish()
        self.index.update(finished_at=self.dependencies.clock.now(), duration_seconds=self.dependencies.clock.elapsed() - began)
        report = self.save()
        self.dependencies.stop.seal()
        if self.index["interrupted"] != self.dependencies.stop.interrupted:
            report = self.save()
        failed = any(entry["error"] or entry["storage_error"] or entry["artifact_errors"]
                     or entry["state"] != COMPLETED for entry in self.entries)
        return SeriesResult(self.dependencies.store.directory, report, 130 if self.dependencies.stop.interrupted else int(failed))

    def run(self) -> SeriesResult:
        began = self.dependencies.clock.elapsed()
        self._initialize()
        with Failure() as failure:
            self._run_records()
        if failure.error is not None:
            self.dependencies.stop.request()
            with Failure():
                self._drain()
            with Failure():
                _ = self._finish(began)
            failure.reraise()
        return self._finish(began)
