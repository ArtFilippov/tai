"""Transient run journal and measurements of FITD candidate text."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from difflib import SequenceMatcher
from typing import override

from journal.observation import Observation


class MemoryJournal(Observation):
    """Retain one run's events without a filesystem-backed journal."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[dict[str, object]] = []

    @override
    def schema(self) -> dict[str, str | int]:
        return {
            "schema_id": "tai.memory",
            "schema_version": 1
        }

    @override
    def record(self, event: dict[str, object]) -> None:
        self.entries.append(
            deepcopy(event)
        )

    @override
    def events(self) -> Sequence[dict[str, object]]:
        return deepcopy(self.entries)

    @override
    def report(self) -> Mapping[str, object]:
        """Compare generated candidates and list requests actually sent to Target.

        Returns:
            Mapping[str, object]: Candidate changes, distinct requests and outcome.

        Raises:
            ValueError: If the run has no recorded base request.
        """
        for event in self.entries:
            match event:
                case {"event": "start", "base": str() as base}:
                    return {
                        "base": base,
                        "candidate_changes": self.changes(base),
                        "target_requests": self.requests(),
                        **self.outcome()
                    }
                case _:
                    continue
        raise ValueError("Run journal has no base request")

    @override
    def changes(self, base: str) -> list[dict[str, object]]:
        """Measure each generated candidate against its preceding text.

        Args:
            base (str): Original request preceding the first candidate.

        Returns:
            list[dict[str, object]]: Ordered character-similarity measurements.
        """
        previous = base
        changes: list[dict[str, object]] = []
        for event in self.entries:
            match event:
                case {"event": "call", "role": "attacker", "prompt": str() as prompt, "response": str() as text} if self.candidate(prompt):
                    changes.append(
                        {
                            "before": previous,
                            "after": text,
                            "changed": text != previous,
                            "similarity": SequenceMatcher(
                                None, previous, text
                            ).ratio()
                        }
                    )
                    previous = text
                case _:
                    continue
        return changes

    @override
    def requests(self) -> list[str]:
        """List unique requests actually transmitted to Target, in order.

        Returns:
            list[str]: Distinct Target requests.
        """
        requests: list[str] = []
        for event in self.entries:
            match event:
                case {"event": "call", "role": "target", "prompt": str() as prompt} if prompt not in requests:
                    requests.append(prompt)
                case _:
                    continue
        return requests

    @override
    def outcome(self) -> Mapping[str, object]:
        """Keep presented text and the judge's verdict separate from candidates.

        Returns:
            Mapping[str, object]: Recorded result, verdict and run status.
        """
        result: str | None = None
        verdict: bool | None = None
        status: str | None = None
        for event in self.entries:
            match event:
                case {"event": "result", "text": str() as text}:
                    result = text
                    verdict = None
                case {"event": "judgment", "assessment": bool() as assessment}:
                    verdict = assessment
                case {"event": "end", "status": str() as outcome}:
                    status = outcome
                case _:
                    continue
        return {
            "presented_result": result,
            "verdict": verdict,
            "status": status
        }

    @override
    def candidate(self, prompt: str) -> bool:
        """Recognize the pinned FITD generation templates, excluding its checks.

        Args:
            prompt (str): Recorded Attacker instruction.

        Returns:
            bool: Whether FITD requested a new candidate sentence.
        """
        return prompt.startswith(
            (
                "Polish this sentence: ", "You are a helpful assistant.\nYour task is to transform", "You will be given two sentences."
            )
        )
