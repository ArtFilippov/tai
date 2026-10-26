"""Shared execution boundary for finite independent candidate conversations."""

from collections.abc import Iterable, Mapping
from typing import override

from garak.attempt import Conversation, Message, Turn

from attack.assessment import Assessment
from attack.candidate_run import CandidateRun
from journal.journal import Journal
from target.chat import Chat


class AssessedCandidates(CandidateRun):
    """Journal every independent Target exchange before assessing its full reply."""

    def __init__(self, journal: Journal) -> None:
        """Bind the event sink for this run.

        Args:
            journal (Journal): Independent run journal.
        """
        super().__init__()
        self.journal = journal

    @override
    def execute(self, candidates: Iterable[str], target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        text: str | None = None
        for index, prompt in enumerate(
            candidates, start=1
        ):
            response = self.exchange(
                target, prompt, index
            )
            text = response
            if assess(response):
                return {
                    "text": response,
                    "status": "success",
                    "reason": "early_success"
                }
        if text is None:
            return {
                "text": None,
                "status": "no_result",
                "reason": "no_candidates"
            }
        return {
            "text": text,
            "status": "round_limit",
            "reason": "candidates_exhausted"
        }

    def exchange(self, target: Chat, prompt: str, index: int) -> str:
        """Record one complete exchange from a fresh conversation.

        Args:
            target (Chat): Target role.
            prompt (str): Transformed request.
            index (int): One-based candidate position.

        Returns:
            str: Full Target reply.
        """
        conversation = Conversation(
            [
                Turn(
                    "user",
                    Message(prompt)
                )
            ]
        )
        response = target.reply(conversation)
        self.journal.record(
            {
                "event": "call",
                "role": "target",
                "candidate": index,
                "prompt": prompt,
                "response": response
            }
        )
        return response
