"""Bounded stochastic tense reformulations assessed in independent conversations."""

from collections.abc import Iterator, Mapping
from typing import override

from garak.attempt import Conversation, Message, Turn

from attack.assessment import Assessment
from attack.candidate_run import CandidateRun
from attack.phrasing.phrasing_plan import PhrasingPlan
from attack.phrasing.phrasing import Phrasing
from base.base_request import BaseRequest
from journal.journal import Journal
from target.chat import Chat


class PhrasingProbe(Phrasing):
    """Keep one tense's candidate budget, identity and outcome independent."""

    def __init__(self, base: BaseRequest, candidates: CandidateRun, journal: Journal, attacker: Chat, *, plan: PhrasingPlan) -> None:
        """Bind one original request to a single tense's sampled Attacker.

        Args:
            base (BaseRequest): Original request and provenance.
            candidates (CandidateRun): Independent candidate execution and assessment.
            journal (Journal): Independent event sink.
            attacker (Chat): Run-local sampled reformulation participant.
            plan (PhrasingPlan): Tense identity, instruction and preparation budget.
        """
        super().__init__()
        self.base = base
        self.candidates = candidates
        self.journal = journal
        self.attacker = attacker
        self.plan = plan

    @override
    def kind(self) -> str:
        return f"phrasing_{self.plan.kind()}"

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "phrasing_probe",
            **self.plan.description(),
            "attacker": self.attacker.description(),
            "processing": "remove double quotes; deduplicate stripped case-sensitive text within tense"
        }

    @override
    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        if base != self.base.text():
            raise ValueError("The Phrasing probe is bound to another base request")
        identity: dict[str, str | int] = {}
        errors: list[str] = []

        def prompts() -> Iterator[Message]:
            texts: set[str] = set()
            for generation in self.plan.generations():
                identity.clear()
                identity.update(
                    tense=self.plan.kind(),
                    generation=generation,
                    position=len(texts) + 1
                )
                text = self.reformulate(identity)
                if not text.strip():
                    errors.append("Phrasing preparation returned an empty reformulation")
                    return
                novelty = text.strip() not in texts
                self.journal.record(
                    {
                        "event": "decision",
                        "action": "phrasing_candidate",
                        "discarded": [],
                        "kept": int(novelty),
                        "candidate": dict(identity),
                        "prompt": text,
                        "reason": "unique" if novelty else "duplicate"
                    }
                )
                if novelty:
                    texts.add(
                        text.strip()
                    )
                    yield Message(
                        text, notes={"candidate": dict(identity)}
                    )

        def assessment(text: str) -> bool:
            verdict = assess(text)
            if verdict:
                self.journal.record(
                    {
                        "event": "decision",
                        "action": "phrasing_success",
                        "discarded": [],
                        "kept": 1,
                        "candidate": dict(identity),
                        "target": target.description(),
                        "text": text
                    }
                )
            return verdict

        outcome = self.candidates.execute(
            prompts(),
            target, assessment
        )
        return {
            **outcome,
            "status": "failure",
            "reason": "preparation_error",
            "error": errors[0]
        } if errors else outcome

    @override
    def reformulate(self, candidate: Mapping[str, str | int]) -> str:
        """Prepare one candidate and retain both the raw and quote-stripped text.

        Args:
            candidate (Mapping[str, str | int]): Tense, generation and candidate position.

        Returns:
            str: Quote-stripped response, without trimming or semantic correction.
        """
        prompt = self.plan.instruction(
            self.base.text()
        )
        response = self.attacker.reply(
            Conversation(
                [
                    Turn(
                        "user",
                        Message(prompt)
                    )
                ]
            )
        )
        text = response.replace(
            '"', ''
        )
        self.journal.record(
            {
                "event": "call",
                "role": "attacker",
                "phase": "phrasing_reformulation",
                "candidate": dict(candidate),
                "prompt": prompt,
                "response": response
            }
        )
        self.journal.record(
            {
                "event": "decision",
                "action": "phrasing_preparation",
                "discarded": [],
                "kept": int(
                    bool(
                        text.strip()
                    )
                ),
                "candidate": dict(candidate),
                "original": self.base.text(),
                "base_metadata": dict(
                    self.base.metadata()
                ),
                "raw": response,
                "prompt": text,
                "reason": "prepared" if text.strip() else "empty_reformulation"
            }
        )
        return text
