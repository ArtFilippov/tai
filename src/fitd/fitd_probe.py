"""The pinned garak FITD applied to a supplied base request."""

from collections.abc import Iterable
from typing import override

from garak.attempt import Attempt, Conversation, Message, Turn
from garak.probes.fitd import FITD

from fitd.chat import Chat
from fitd.journal import Journal


class FITDProbe(FITD):
    """A single-goal FITD whose model calls and steps belong to this run."""

    def __init__(self, base: str, attacker: Chat, journal: Journal, turns: int) -> None:
        """Bind a supplied goal to isolated roles for one native FITD run.

        Args:
            base (str): Request to reach.
            attacker (Chat): Role generating and checking FITD turns.
            journal (Journal): Destination for observed steps and calls.
            turns (int): Maximum number of target exchanges.
        """
        self.base = base
        self.attacker = attacker
        self.journal = journal
        super().__init__()
        self.level_queries_per_goal: dict[str, list[str]] = {}
        self.num_level_queries = 3
        self.max_calls_per_conv = turns
        self.follow_prompt_cap = False

    @override
    def _get_red_team_response(self, prompt: str) -> str:
        response = self.attacker.reply(
            Conversation(
                [
                    Turn(
                        "user",
                        Message(text=prompt)
                    )
                ]
            )
        )
        self.journal.record(
            {
                "event": "call",
                "role": "attacker",
                "prompt": prompt,
                "response": response
            }
        )
        return response

    @override
    def _create_init_attempts(self) -> Iterable[Attempt]:
        self.set_up_refusal_detector()
        if (opening := self._generate_init_turn(self.base)) is None:
            return []
        self.level_queries_per_goal[self.base] = list(
            self._get_level_queries(
                self.base, opening
            )
        )
        attempt = self._create_attempt(opening)
        attempt.notes.update(
            {
                "turn_num": 0,
                "target_turn": self.base,
                "levels_completed": 0,
                "is_adversarial": False,
                "is_last_turn": False,
                "last_non_refused_prompt": None
            }
        )
        return [attempt]

    @override
    def _execute_all(self, attempts: Iterable[Attempt]) -> Iterable[Attempt]:
        completed: list[Attempt] = []
        for attempt in attempts:
            conversation = attempt.prompt
            prompt = conversation.last_message("user").text
            response = self.generator.reply(conversation)
            self.journal.record(
                {
                    "event": "call",
                    "role": "target",
                    "prompt": prompt,
                    "response": response
                }
            )
            attempt.outputs = [
                Message(
                    text=response, lang="en"
                )
            ]
            self.journal.record(
                {
                    "event": "step",
                    "prompt": prompt,
                    "response": response,
                    "accepted": not self._is_refusal(response),
                    "notes": attempt.notes,
                    "turns": [
                        {
                            "role": turn.role,
                            "text": turn.content.text
                        }
                        for turn in attempt.conversations[0].turns
                    ]
                }
            )
            completed.append(
                self._postprocess_attempt(attempt)
            )
        return completed

    def present(self, attempts: Iterable[Attempt]) -> str | None:
        """Select the last non-refused Target response from completed exchanges.

        Args:
            attempts (Iterable[Attempt]): Exchanges returned by the pinned FITD.

        Returns:
            str | None: The last accepted text, if any.
        """
        text: str | None = None
        for attempt in attempts:
            for output in attempt.outputs:
                if not self._is_refusal(output.text):
                    text = output.text
        return text
