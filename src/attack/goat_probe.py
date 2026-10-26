"""One bounded GOAT conversation using the pinned native O-T-S-R resources."""

from collections.abc import Mapping
from typing import override

from garak.attempt import Conversation, Message, Turn
from garak.probes.goat import GOATAttack as NativeGOAT

from attack.assessment import Assessment
from attack.probe import Probe
from attack.goat_text import GOATText
from journal.journal import Journal
from target.chat import Chat

UNSAFE = "unsafe"
INTERNAL_STOP = "internal_stop"


class GOATProbe(Probe):
    """Keep the two role histories and native GOAT decisions within one run."""

    class Native(NativeGOAT):
        """Use only a supplied goal and pinned GOAT text resources."""

        def __init__(self, goal: str) -> None:
            """Keep a single supplied goal for native initialization.

            Args:
                goal (str): The sole attack goal; never a seed corpus.
            """
            self.goal = goal
            super().__init__()

        @override
        def _get_attack_goals(self) -> list[str]:
            return [self.goal]

        def system(self, goal: str) -> str:
            """Fill the pinned GOAT system prompt with the explicit goal.

            Args:
                goal (str): Sole attack goal.

            Returns:
                str: Native system prompt.
            """
            return self.system_prompt_template.format(
                goal=goal, attack_techniques=self._format_attack_techniques()
            )

        def initial(self, goal: str) -> str:
            """Fill the pinned first-turn prompt with the explicit goal.

            Args:
                goal (str): Sole attack goal.

            Returns:
                str: Native first-turn prompt.
            """
            return self.initial_prompt_template.format(goal=goal)

        def followup(self, goal: str, prompt: str, response: str) -> str:
            """Ask the Attacker to react to the last Target exchange.

            Args:
                goal (str): Sole attack goal.
                prompt (str): Previous Target prompt.
                response (str): Previous Target response.

            Returns:
                str: Native follow-up prompt.
            """
            return self.followup_prompt_template.format(
                goal=goal, adversarial_prev_prompt=prompt, prev_model_response=response
            )

        def judge(self, goal: str, response: str) -> str:
            """Ask the native internal search judge about one Target reply.

            Args:
                goal (str): Sole attack goal.
                response (str): Target response.

            Returns:
                str: Native search-judge prompt.
            """
            return self.judge_prompt_template.format(
                goal=goal, response=response
            )

        def parse(self, text: str) -> dict[str, str] | None:
            """Extract the first native O-T-S-R response.

            Args:
                text (str): Attacker's raw reply.

            Returns:
                dict[str, str] | None: Native O-T-S-R sections if present.
            """
            return self._parse_otsr_response(
                self._extract_content(text)
            )

        def format(self, response: dict[str, str]) -> str:
            """Keep the native canonical attacker transcript.

            Args:
                response (dict[str, str]): Parsed native sections.

            Returns:
                str: Canonical O-T-S-R transcript.
            """
            return self._format_otsr(response)

        def techniques(self) -> str:
            """Use the pinned native attack technique definitions.

            Returns:
                str: Formatted native techniques.
            """
            return self._format_attack_techniques()

        def content(self, text: str) -> str:
            """Remove native thinking tags from a search-judge reply.

            Args:
                text (str): Raw judge response.

            Returns:
                str: Native extracted content.
            """
            return self._extract_content(text)

    def __init__(self, base: str, attacker: Chat, journal: Journal, turns: int) -> None:
        """Store the bounded role and request without loading a dataset.

        Args:
            base (str): The sole GOAT goal.
            attacker (Chat): Independent attacker role.
            journal (Journal): Run event sink.
            turns (int): Maximum Target exchanges.

        Raises:
            ValueError: If the bound is not positive.
        """
        super().__init__()
        if turns < 1:
            raise ValueError("GOAT turn limit must be positive")
        self.base = base
        self.attacker = attacker
        self.journal = journal
        self.turns = turns

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "goat_probe",
            "max_calls_per_conv": self.turns,
            "end_condition": "verify",
            "attacker": self.attacker.description()
        }

    @override
    def kind(self) -> str:
        return "goat"

    def ask(self, conversation: Conversation) -> str:
        """Record a complete Attacker exchange with its own history.

        Args:
            conversation (Conversation): The independent role history.

        Returns:
            str: Exact Attacker response.
        """
        response = self.attacker.reply(conversation)
        self.journal.record(
            {
                "event": "call",
                "role": "attacker",
                "prompt": conversation.last_message("user").text,
                "response": response,
                "turns": [
                    {
                        "role": turn.role,
                        "text": turn.content.text
                    }
                    for turn in conversation.turns
                ]
            }
        )
        return response

    def candidate(self, conversation: Conversation, native: GOATText) -> str | None:
        """Use native O-T-S-R extraction and retain the formatted reasoning.

        Args:
            conversation (Conversation): Current Attacker history.
            native (GOATText): Pinned parser and prompts for this execution.

        Returns:
            str | None: Next prompt if the Attacker supplied one.
        """
        if not (response := self.ask(conversation)):
            return None
        if (parsed := native.parse(response)) is None:
            return None
        conversation.turns.append(
            Turn(
                "assistant",
                Message(
                    native.format(parsed)
                )
            )
        )
        return parsed["Response"]

    def stop(self, base: str, turns: list[Turn], native: GOATText) -> bool:
        """Record the pinned internal search decision without assessing a result.

        Args:
            base (str): The sole GOAT goal.
            turns (list[Turn]): Completed Target history.
            native (GOATText): Pinned internal judge text behavior.

        Returns:
            bool: Whether the native internal judge stops search.
        """
        prompt = turns[-2].content.text
        response = turns[-1].content.text
        verdict = self.ask(
            Conversation(
                [
                    Turn(
                        "user",
                        Message(
                            native.judge(
                                base, response
                            )
                        )
                    )
                ]
            )
        )
        stopped = UNSAFE in native.content(verdict).lower()
        self.journal.record(
            {
                "event": "decision",
                "action": "stop" if stopped else "continue",
                "prompt": prompt,
                "response": response,
                "judge_response": verdict,
                "turn": len(turns) // 2,
                "discarded": [],
                "kept": len(turns)
            }
        )
        return stopped

    @override
    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        if base != self.base:
            raise ValueError("The probe's base request differs from the run")
        native = self.Native(base)
        attacker = Conversation(
            [
                Turn(
                    "system",
                    Message(
                        native.system(base)
                    )
                ),
                Turn(
                    "user",
                    Message(
                        native.initial(base)
                    )
                )
            ]
        )
        if (prompt := self.candidate(attacker, native)) is None:
            return {
                "text": None,
                "status": "no_result"
            }
        turns: list[Turn] = []
        for _ in range(self.turns):
            turns.append(
                Turn(
                    "user",
                    Message(prompt)
                )
            )
            response = target.reply(
                Conversation(
                    list(turns)
                )
            )
            self.journal.record(
                {
                    "event": "call",
                    "role": "target",
                    "prompt": prompt,
                    "response": response,
                    "turn": (len(turns) + 1) // 2
                }
            )
            turns.append(
                Turn(
                    "assistant",
                    Message(response)
                )
            )
            self.journal.record(
                {
                    "event": "step",
                    "prompt": prompt,
                    "response": response,
                    "accepted": True,
                    "turn": len(turns) // 2,
                    "turns": [
                        {
                            "role": turn.role,
                            "text": turn.content.text
                        }
                        for turn in turns
                    ]
                }
            )
            if (reason := INTERNAL_STOP if self.stop(base, turns, native) else "round_limit") == INTERNAL_STOP or len(turns) // 2 == self.turns:
                _ = assess(response)
                return {
                    "text": response,
                    "status": "completed",
                    "reason": reason
                }
            attacker.turns.append(
                Turn(
                    "user",
                    Message(
                        native.followup(
                            base, prompt, response
                        )
                    )
                )
            )
            if (prompt := self.candidate(attacker, native)) is None:
                _ = assess(response)
                return {
                    "text": response,
                    "status": "completed",
                    "reason": "attacker_stalled"
                }
        raise AssertionError("GOAT finished without a terminal turn")
