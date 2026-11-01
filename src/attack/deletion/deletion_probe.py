"""Native pinned garak deletion variants for an explicit request."""

from collections.abc import Iterator, Mapping
from typing import override

from garak.attempt import Message

from attack.assessment import Assessment
from attack.candidate_run import CandidateRun
from attack.deletion.deletion_limits import DeletionLimits
from attack.deletion.deletion_variants import DeletionVariants
from attack.probe import Probe
from journal.journal import Journal
from target.chat import Chat


class DeletionProbe(Probe):
    """Generate native backspace variants without loading the garak corpus."""

    def __init__(self, base: str, candidates: CandidateRun, journal: Journal, *, budget: DeletionLimits, variants: DeletionVariants) -> None:
        """Bind an original request to one independent candidate run.

        Args:
            base (str): Exact original request.
            candidates (CandidateRun): Independent Target execution.
            journal (Journal): Transformation provenance sink.
            budget (DeletionLimits): Native generation limits.
            variants (DeletionVariants): Native garak generator.
        """
        super().__init__()
        self.base = base
        self.candidates = candidates
        self.journal = journal
        self.variants = variants
        self.budget = budget

    @override
    def kind(self) -> str:
        return "bad_characters_deletion"

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "deletion_probe",
            "source": "garak.probes.badchars.BadCharacters._generate_deletion_variants",
            "revision": "3f50ea5ff9cd7050099940647c15c39b07a93392",
            "budget": {
                "perturbations": self.budget.perturbations,
                "positions": self.budget.positions,
                "ascii": self.budget.ascii
            }
        }

    def prompts(self) -> Iterator[Message]:
        """Journal each native transformation immediately before transmission.

        Yields:
            Message: A backspace request with its native transformation details.
        """
        for index, variant in enumerate(
            self.variants.generate(self.base),
            start=1
        ):
            self.journal.record(
                {
                    "event": "decision",
                    "action": "deletion",
                    "discarded": [],
                    "kept": 1,
                    "candidate": index,
                    "prompt": variant["text"],
                    "bad_character_category": "deletion",
                    "perturbation_count": variant["count"],
                    "operations": variant["operations"]
                }
            )
            yield Message(
                str(variant["text"]),
                notes={
                    "bad_character_category": "deletion",
                    "perturbation_count": variant["count"],
                    "operations": variant["operations"]
                }
            )

    @override
    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        if base != self.base:
            raise ValueError("The probe's base request differs from the run")
        self.budget.validate()
        return self.candidates.execute(
            self.prompts(),
            target, assess
        )
