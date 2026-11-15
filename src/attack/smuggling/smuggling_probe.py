"""One deterministic homoglyph candidate for the caller's original request."""

from collections.abc import Mapping, Sequence
from random import Random
from typing import override

from garak.attempt import Message
from garak.probes.smuggling import DEFAULT_HOMOGLYPH_MAP

from attack.assessment import Assessment
from attack.candidate_run import CandidateRun
from attack.probe import Probe
from base.base_request import BaseRequest
from journal.journal import Journal
from target.chat import Chat


class SmugglingProbe(Probe):
    """Preserve the original request while presenting its native homoglyph form."""

    def __init__(self, base: BaseRequest, candidates: CandidateRun, journal: Journal, *,
                 techniques: Sequence[str] = ("homoglyph",)) -> None:
        """Bind one request and the independently assessed candidate execution.

        Args:
            base (BaseRequest): Original request and provenance.
            candidates (CandidateRun): Independent one-exchange Target execution.
            journal (Journal): Run-local transformation and outcome sink.
            techniques (Sequence[str], default=("homoglyph",)): Nonempty homoglyph-only selection.

        Raises:
            ValueError: If selection is empty or includes unsupported techniques.
        """
        super().__init__()
        if not techniques or set(techniques) != {"homoglyph"}:
            raise ValueError("Smuggling techniques must be a nonempty subset of homoglyph")
        self.base = base
        self.candidates = candidates
        self.journal = journal

    @override
    def kind(self) -> str:
        return "smuggling"

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "smuggling_probe",
            "techniques": ["homoglyph"],
            "source": "garak.probes.smuggling._homoglyph_replace",
            "revision": "3f50ea5ff9cd7050099940647c15c39b07a93392",
            "settings": {
                "seed": 0,
                "candidates": 1,
                "replacement": "every eligible character",
                "mapping": DEFAULT_HOMOGLYPH_MAP
            }
        }

    @override
    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        if base != self.base.text():
            raise ValueError("The Smuggling probe is bound to another base request")
        random = Random(0)
        text = "".join(random.choice(DEFAULT_HOMOGLYPH_MAP[character]) if character in DEFAULT_HOMOGLYPH_MAP else character for character in base)
        identity = {
            "technique": "homoglyph",
            "position": 1,
            "seed": 0
        }
        self.journal.record(
            {
                "event": "decision",
                "action": "smuggling_transformation",
                "discarded": [],
                "kept": int(text != base),
                "candidate": identity,
                "original": base,
                "base_metadata": dict(
                    self.base.metadata()
                ),
                "prompt": text,
                "reason": "transformed" if text != base else "unchanged_text"
            }
        )

        def assessment(text: str) -> bool:
            verdict = assess(text)
            if verdict:
                self.journal.record(
                    {
                        "event": "decision",
                        "action": "smuggling_success",
                        "discarded": [],
                        "kept": 1,
                        "candidate": identity,
                        "target": target.description(),
                        "text": text
                    }
                )
            return verdict

        return self.candidates.execute(
            [Message(text, notes={"candidate": identity})] if text != base else [],
            target, assessment
        )
