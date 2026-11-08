"""One request-bound BEAST attempt with independent final assessment."""

from collections.abc import Mapping
from typing import override

from garak.attempt import Conversation, Message, Turn
from garak.resources.common import REJECTION_STRINGS

from attack.assessment import Assessment
from attack.beast.beast_search import BeastSearch
from attack.desired_response.desired_response import DesiredResponse
from attack.probe import Probe
from journal.journal import Journal
from target.chat import Chat


class BeastProbe(Probe):
    """Search locally, then send only the best suffix to Target once."""

    def __init__(self, base: str, source: DesiredResponse, search: BeastSearch, journal: Journal) -> None:
        """Bind an answer source and search to exactly one request.

        Args:
            base (str): Exact base request.
            source (DesiredResponse): Provided or deferred answer.
            search (BeastSearch): Bounded local token search.
            journal (Journal): Attempt's event sink.
        """
        super().__init__()
        self.base = base
        self.source = source
        self.search = search
        self.journal = journal

    @override
    def kind(self) -> str:
        return "beast"

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "beast_probe",
            "source": self.source.description(),
            **self.search.description(),
            "revision": "3f50ea5ff9cd7050099940647c15c39b07a93392",
            "adaptation": (
                "request-bound BEAST: search/final user-only chat; minimize desired-answer loss; count first token and extend each beam's own suffix; "
                "continue trials; full-response refusal diagnostic; independently judge only the final Target reply; no seed or stop_early"
            )
        }

    @override
    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        if base != self.base:
            raise ValueError("BEAST answer belongs to another base request")
        answer = self.source.response(
            base, self.journal
        )
        if not answer.strip():
            raise ValueError("BEAST desired response is empty")
        candidate = self.search.search(
            base, answer, self.journal
        )
        if candidate is None:
            self.journal.record(
                {
                    "event": "decision",
                    "action": "beast_exhausted",
                    "discarded": [],
                    "kept": 0
                }
            )
            return {
                "text": None,
                "status": "no_result",
                "reason": "search_exhausted"
            }
        suffix = self.search.decode(candidate)
        prompt = base + suffix
        self.journal.record(
            {
                "event": "decision",
                "action": "beast_selection",
                "discarded": [],
                "kept": 1,
                "suffix": suffix,
                "score": self.search.score(),
                "objective": "mean_desired_response_cross_entropy"
            }
        )
        response = target.reply(
            Conversation(
                [
                    Turn(
                        "user",
                        Message(prompt)
                    )
                ]
            )
        )
        self.journal.record(
            {
                "event": "call",
                "role": "target",
                "prompt": prompt,
                "response": response
            }
        )
        self.journal.record(
            {
                "event": "decision",
                "action": "beast_refusal_signal",
                "discarded": [],
                "kept": 1,
                "non_refusal": not any(( phrase.casefold() in response.casefold() for phrase in REJECTION_STRINGS ))
            }
        )
        _ = assess(response)
        return {
            "text": response,
            "status": "completed",
            "reason": "final_assessment"
        }
