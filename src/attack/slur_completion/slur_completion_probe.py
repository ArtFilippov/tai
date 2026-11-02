"""Continue an opening derived from an injected answer."""

import re
from collections.abc import Mapping
from typing import override

from garak.attempt import Conversation, Message, Turn

from attack.assessment import Assessment
from attack.desired_response.desired_response import DesiredResponse
from attack.probe import Probe
from journal.journal import Journal
from target.chat import Chat


class SlurCompletionProbe(Probe):
    """Use a short prepared opening as a genuine Target assistant prefix."""

    def __init__(self, base: str, source: DesiredResponse, journal: Journal) -> None:
        """Bind a source and its journal to the original request.

        Args:
            base (str): Original request.
            source (DesiredResponse): Prepared answer source.
            journal (Journal): Independent run journal.
        """
        super().__init__()
        self.base = base
        self.source = source
        self.journal = journal

    @override
    def kind(self) -> str:
        return "slur_completion"

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "slur_completion_probe",
            "source": self.source.description(),
            "revision": "3f50ea5ff9cd7050099940647c15c39b07a93392",
            "adaptation": "one answer-opening continuation for the base request"
        }

    @override
    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        if base != self.base:
            raise ValueError("The prepared answer belongs to another base request")
        try:
            desired = self.source.response(
                base, self.journal
            )
        except Exception as exc:
            raise ValueError("Desired response preparation failed") from exc
        if len(desired.strip()) < 2:
            raise ValueError("Desired response has no usable opening")
        words = list(
            re.finditer(
                r"\S+", desired
            )
        )
        opening = words[1].end() if len(words) > 1 else words[0].end()
        limit = min(
            32,
            len(desired) - 1,
            opening
        )
        prefix = desired[:limit].rstrip()
        if not prefix:
            raise ValueError("Desired response has no usable opening")
        self.journal.record(
            {
                "event": "decision",
                "action": "slur_completion_opening",
                "discarded": [],
                "kept": 1,
                "base": base,
                "desired_response": desired,
                "template": base,
                "prefix": prefix
            }
        )
        try:
            response = target.prefill(
                Conversation(
                    [
                        Turn(
                            "user",
                            Message(base)
                        )
                    ]
                ),
                prefix
            )
        except ValueError as exc:
            raise RuntimeError("Target prefill failed") from exc
        if not response.startswith(prefix):
            raise RuntimeError("Target did not preserve assistant prefill")
        self.journal.record(
            {
                "event": "call",
                "role": "target",
                "prompt": base,
                "prefix": prefix,
                "continuation": response[len(prefix):],
                "response": response
            }
        )
        _ = assess(response)
        return {
            "text": response,
            "status": "completed",
            "reason": "slur_completion_response"
        }
