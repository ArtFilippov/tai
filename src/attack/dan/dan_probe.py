"""Two-exchange DAN adaptation independent of any prompt catalog."""

from collections.abc import Mapping
from typing import override

from garak.attempt import Conversation, Message, Turn

from attack.assessment import Assessment
from attack.probe import Probe
from base.base_request import BaseRequest
from journal.journal import Journal
from prompt_collection.prompt_collection import PromptCollection
from target.chat import Chat


class DanProbe(Probe):
    """Continue every completed activation with the unchanged original request."""

    def __init__(self, base: BaseRequest, prompts: PromptCollection, journal: Journal) -> None:
        """Bind the original request, abstract collection and run journal.

        Args:
            base (BaseRequest): Exact request with its provenance.
            prompts (PromptCollection): Rendered activation candidates in traversal order.
            journal (Journal): Independent attempt journal.
        """
        super().__init__()
        self.base = base
        self.prompts = prompts
        self.journal = journal

    @override
    def kind(self) -> str:
        return "dan"

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "dan_probe",
            "collection": self.prompts.description(),
            "adaptation": "activation followed by exact base request; assess full second reply",
        }

    @override
    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        def exchange(conversation: Conversation, *, candidate: Mapping[str, str | int], phase: str) -> str:
            """Save a completed exchange before continuing the candidate.

            Args:
                conversation (Conversation): Complete candidate history.
                candidate (Mapping[str, str | int]): Source identity and traversal position.
                phase (str): Activation or base-request phase.

            Returns:
                str: Complete Target reply.
            """
            text = target.reply(conversation)
            self.journal.record(
                {
                    "event": "call",
                    "role": "target",
                    "phase": phase,
                    "candidate": dict(candidate),
                    "prompt": conversation.turns[-1].content.text,
                    "metadata": conversation.turns[-1].content.notes,
                    "response": text
                }
            )
            return text

        if base != self.base.text():
            raise ValueError("The DAN collection is bound to another base request")
        identity = target.description()
        text: str | None = None
        for position, prompt in enumerate(
            self.prompts.candidates(),
            start=1
        ):
            candidate = {
                **prompt.identity(),
                "position": position
            }
            turns = [
                Turn(
                    "user",
                    Message(
                        prompt.text()
                    )
                )
            ]
            activation = exchange(
                Conversation(turns),
                candidate=candidate, phase="activation"
            )
            conversation = Conversation(
                [
                    *turns,
                    Turn(
                        "assistant",
                        Message(activation)
                    ),
                    Turn(
                        "user",
                        Message(
                            base,
                            notes=dict(
                                self.base.metadata()
                            )
                        )
                    )
                ]
            )
            text = exchange(
                conversation, candidate=candidate, phase="base_request"
            )
            self.journal.record(
                {
                    "event": "decision",
                    "action": "dan_candidate",
                    "discarded": [],
                    "kept": 1,
                    "candidate": candidate,
                    "text": text
                }
            )
            if assess(text):
                self.journal.record(
                    {
                        "event": "decision",
                        "action": "dan_success",
                        "discarded": [],
                        "kept": 1,
                        "candidate": candidate,
                        "target": identity,
                        "text": text
                    }
                )
                return {
                    "text": text,
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
