"""Public Python assembly of past/future attacks over shared role resources."""

from pathlib import Path

from attack.attack_setup import AttackSetup
from attack.phrasing.configured_phrasing_plan import ConfiguredPhrasingPlan
from attack.phrasing.envelopes.phrasing_attack_envelope import PhrasingAttackEnvelope
from attack.phrasing.journaled_phrasing_pair import JournaledPhrasingPair
from attack.phrasing.phrasing_pair_envelope import PhrasingPairEnvelope
from base.base_request import BaseRequest
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal


class DiskPhrasingPairEnvelope(PhrasingPairEnvelope):
    """Compose two journaled outcomes without allocating separate model weights."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, directory: Path, *, attempts: int = 20) -> None:
        """Configure both tenses without accessing models or reserving files.

        Args:
            base (BaseRequest): Original request and provenance.
            setup (AttackSetup): Shared independent role resources.
            directory (Path): New directory for two journals.
            attempts (int, default=20): Maximum preparation calls per tense.
        """
        super().__init__(
            JournaledPhrasingPair(
                directory,
                [
                    ConfiguredPhrasingPlan(
                        "past", attempts
                    ),
                    ConfiguredPhrasingPlan(
                        "future", attempts
                    )
                ],
                lambda plan, path: PhrasingAttackEnvelope(base, setup, VersionedJournal(DiskJournal(path)), plan=plan), lambda path: ParsedJournal(path)
            )
        )
