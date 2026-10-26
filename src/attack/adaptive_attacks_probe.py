"""Pinned garak adaptive template and bundled suffixes for one supplied goal."""

from collections.abc import Mapping
from typing import override

from garak.probes.adaptive_attacks import ADAPTIVE_TEMPLATE, SUFFIXES_PATH

from attack.assessment import Assessment
from attack.candidate_run import CandidateRun
from attack.probe import Probe
from target.chat import Chat

SUFFIX_COUNT = 5


class AdaptiveAttacksProbe(Probe):
    """Compose the native template with the explicit base, never garak's seeds."""

    def __init__(self, base: str, candidates: CandidateRun) -> None:
        """Bind a goal and independent candidate execution to one attempt.

        Args:
            base (str): Original request.
            candidates (CandidateRun): Journaled candidate execution.
        """
        super().__init__()
        self.base = base
        self.candidates = candidates

    @override
    def kind(self) -> str:
        return "adaptive_attacks"

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "adaptive_attacks_probe",
            "source": "garak.probes.adaptive_attacks",
            "revision": "3f50ea5ff9cd7050099940647c15c39b07a93392",
            "candidates": SUFFIX_COUNT
        }

    @override
    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        if base != self.base:
            raise ValueError("The probe's base request differs from the run")
        with SUFFIXES_PATH.open(encoding="utf-8") as stream:
            suffixes = [line.strip() for line in stream if line.strip()]
        if len(suffixes) != SUFFIX_COUNT:
            raise ValueError("The pinned adaptive suffix resource must contain five candidates")
        return self.candidates.execute(
            (ADAPTIVE_TEMPLATE.format(goal=base, suffix=suffix) for suffix in suffixes),
            target, assess
        )
