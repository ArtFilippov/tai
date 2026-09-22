"""One admission and its externally saved outcome, independent of completion order."""

from dataclasses import dataclass

from core.primitives.messages import Message
from core.runs import RunSnapshot

from .contracts import RunHandle
from .failure_capture import Failure
from .series_contracts import SeriesDependencies, SeriesIdentity, snapshot_outcome


def failure_data(error: Exception, phase: str, dependencies: SeriesDependencies) -> dict[str, str]:
    return {"phase": phase, "type": type(error).__name__, "message": dependencies.redact(str(error))}


@dataclass
class Attempt:
    identity: SeriesIdentity
    base: Message
    start_time: float
    handle: RunHandle | None = None
    phase: str = "create_experiment"
    error: dict[str, str] | None = None

    def admit(self, dependencies: SeriesDependencies) -> bool:
        with Failure() as failure:
            self.handle = dependencies.create(self.identity, self.base)
        if failure.error is not None:
            self.error = failure_data(failure.error, self.phase, dependencies)
            return True
        if dependencies.stop.interrupted:
            return False
        self._start(dependencies)
        return True

    def _start(self, dependencies: SeriesDependencies) -> None:
        assert self.handle is not None
        self.phase = "start"
        with Failure() as failure:
            self.handle.start()
        if failure.error is not None:
            self.error = failure_data(failure.error, self.phase, dependencies)
            self.stop(dependencies)
        self.phase = "wait"

    def done(self) -> bool:
        return self.handle is None or self.handle.status().run_id is None or self.handle.wait(0)

    def join(self) -> None:
        if self.handle is not None and self.handle.status().run_id is not None:
            _ = self.handle.wait()

    def stop(self, dependencies: SeriesDependencies) -> None:
        if self.handle is not None:
            with Failure() as failure:
                self.handle.stop()
            if failure.error is not None and self.error is None:
                self.error = failure_data(failure.error, "stop", dependencies)

    def active(self) -> dict[str, object]:
        status = self.handle.status() if self.handle is not None else None
        return {**snapshot_outcome(None), "run_id": status.run_id if status is not None else None,
                "state": status.state.value if status is not None else None,
                "phase": "active", "finished_at": None, "duration_seconds": None,
                "result": None, "evaluation": None, "error": self.error, "storage_error": None,
                "artifact_errors": {}, "artifacts": {}}

    def _save(self, dependencies: SeriesDependencies) -> tuple[RunSnapshot | None, dict[str, str] | None]:
        if (handle := self.handle) is None:
            return None, None
        status = handle.status()
        if status.run_id is None:
            return RunSnapshot(status, None), None
        snapshot = handle.snapshot()
        with Failure() as failure:
            handle.save()
        storage_error = failure_data(failure.error, "save", dependencies) if failure.error is not None else None
        if self.error is None and (error := handle.journal().error) is not None:
            self.error = failure_data(error, "journal", dependencies)
        return snapshot, storage_error

    def capture(self, dependencies: SeriesDependencies) -> dict[str, object]:
        snapshot, storage_error = self._save(dependencies)
        outcome = snapshot_outcome(snapshot)
        reason = outcome["reason"]
        outcome.update(dependencies.store.inspect(snapshot), error=self.error, storage_error=storage_error,
                       reason=dependencies.redact(reason) if isinstance(reason, str) else None,
                       phase="finished", finished_at=dependencies.clock.now(),
                       duration_seconds=dependencies.clock.elapsed() - self.start_time)
        return outcome
