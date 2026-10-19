"""The pinned garak FITD applied to a supplied base request."""

from collections.abc import Iterable, Mapping
from typing import override

from garak.attempt import Attempt, Conversation, Message, Turn
from garak.probes.fitd import FITD

from attack.assessment import Assessment
from attack.probe import Probe
from journal.journal import Journal
from target.chat import Chat


class FITDProbe(FITD, Probe):
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
        self.assessments: list[Assessment] = []
        self.success = False
        self.level_queries_per_goal: dict[str, list[str]] = {}
        self.max_calls_per_conv = turns

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
            if not self._is_refusal(response) and self.assessments[0](response):
                self.success = True
                completed.append(attempt)
                break
            completed.append(
                self._postprocess_attempt(attempt)
            )
        return completed

    @override
    def _verify_is_last_response_success(self, attempt: Attempt) -> list[bool]:
        # Keep garak's internal Attacker checks, but let the independent judge
        # decide whether the run stops rather than its unsafe-prefix heuristic.
        return [False for _ in super()._verify_is_last_response_success(attempt)]

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

    @override
    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        if base != self.base:
            raise ValueError("The probe's base request differs from the run")
        self.assessments.append(assess)
        text = self.present(
            self.probe(target)
        )
        if self.success:
            return {
                "text": text,
                "status": "success"
            }
        if text is None:
            return {
                "text": None,
                "status": "no_result"
            }
        return {
            "text": text,
            "status": "round_limit"
        }
