"""Compose controlled participants at the public Phrasing run boundary."""

from pathlib import Path

from attack.assessed_candidates import AssessedCandidates
from attack.attack_run import AttackRun
from attack.phrasing.configured_phrasing_plan import ConfiguredPhrasingPlan
from attack.phrasing.phrasing_probe import PhrasingProbe
from base.base_request import BaseRequest
from journal.disk_journal import DiskJournal
from journal.versioned_journal import VersionedJournal
from judge.judge import Judge
from judge.recorded_judge import RecordedJudge
from target.chat import Chat


def attempt(path: Path, base: BaseRequest, attacker: Chat, target: Chat, *, judge: Judge, tense: str, attempts: int) -> str | None:
    """Run a real journaled attack with explicit scripted role boundaries.

    Args:
        path (Path): Independent journal destination.
        base (BaseRequest): Original request with provenance.
        attacker (Chat): Preparation participant.
        target (Chat): Candidate response participant.
        judge (Judge): Original-request fulfillment assessment.
        tense (str): One independent tense.
        attempts (int): Preparation call budget.

    Returns:
        str | None: Selected full Target reply.
    """
    journal = VersionedJournal(
        DiskJournal(path)
    )
    return AttackRun(
        base, target,
        RecordedJudge(
            judge, journal,
            base.text()
        ),
        journal,
        probe=PhrasingProbe(
            base,
            AssessedCandidates(journal),
            journal, attacker,
            plan=ConfiguredPhrasingPlan(
                tense, attempts
            )
        )
    ).run()
