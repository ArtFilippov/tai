"""Readiness ownership and finalization independent of observer implementations."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from core.chats.reservation_utils import acquire_chats, release_chats
from core.runs import UserState, UserStatus
from core.runs.operations import ResourcesReleased, WorkerOutcome

from .contracts import ReadinessServices
from .failure_capture import Failure
from .profiles import ParticipantProfile
from .readiness import check_roles


@dataclass
class ReadinessSession:
    profile: ParticipantProfile
    services: ReadinessServices
    run_id: str = field(default_factory=lambda: str(uuid4()))
    data: dict[str, object] = field(default_factory=lambda: {"ready": False})
    acquired: bool = False
    failure: Failure = field(default_factory=Failure)
    rendered: str | None = None

    def _check(self) -> None:
        with self.failure:
            self.services.journal.begin(self.run_id)
            acquire_chats(tuple(self.profile.chats().values()), self.services.journal)
            self.acquired = True
            self.data = check_roles(self.profile, self.services, self.run_id)

    def _outcome(self) -> WorkerOutcome:
        released = None
        if self.acquired:
            release_chats(tuple(self.profile.chats().values()), self.services.journal)
            released = ResourcesReleased(self.run_id, tuple(self.profile.chats()))
        error = self.failure.error
        return WorkerOutcome(
            UserStatus(state=UserState.COMPLETED if self.data["ready"] else UserState.FAILED, run_id=self.run_id,
                       progress=self.services.journal.status().progress,
                       reason=f"{type(error).__name__}: {error}" if error else "preflight"),
            None, error, released,
        )

    def finish(self, report: Path | None) -> None:
        outcome = self._outcome()
        with Failure() as failure:
            self.rendered = self.services.journal.finish(outcome, report=(self.data, report) if self.failure.error is None else None)
        if self.failure.error is None and failure.error is not None:
            raise failure.error

    def run(self, report: Path | None) -> tuple[dict[str, object], str]:
        try:
            self._check()
        finally:
            self.finish(report)
        self.failure.reraise()
        assert self.rendered is not None
        return self.data, self.rendered
