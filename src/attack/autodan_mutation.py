"""Adapter for native garak mutation using a separate Attacker conversation."""

from typing import override

from garak.attempt import Conversation, Message

from attack.autodan_mutation_contract import AutoDANMutationContract
from journal.journal import Journal
from target.chat import Chat

ABORT = "AutoDAN mutation failed"


class AutoDANMutation(AutoDANMutationContract):
    """Supply one bounded native mutation request to an independent Attacker."""

    def __init__(self, attacker: Chat, journal: Journal) -> None:
        """Bind the independent role and journal.

        Args:
            attacker (Chat): Independent mutation role.
            journal (Journal): Attempt event sink.
        """
        super().__init__()
        self.attacker = attacker
        self.journal = journal

    @override
    def generate(self, prompt: Conversation) -> list[Message]:
        """Respond once to the native mutation prompt, failing on an empty reply.

        Args:
            prompt (Conversation): Native mutation instructions.

        Returns:
            list[Message]: Single nonempty mutation for garak.

        Raises:
            SystemExit: If the Attacker stalls or fails (caught by the probe).
        """
        try:
            response = self.attacker.reply(prompt)
        except Exception as exc:
            raise SystemExit(ABORT) from exc
        self.journal.record(
            {
                "event": "call",
                "role": "attacker",
                "prompt": prompt.last_message("user").text,
                "response": response
            }
        )
        if not response.strip():
            raise SystemExit(ABORT)
        return [
            Message(response)
        ]

    @override
    def description(self) -> dict[str, object]:
        return self.attacker.description()
