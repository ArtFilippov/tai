"""Pinned DRA disguise and reconstruction for an explicit base request."""

import json
import random
from collections.abc import Iterator, Mapping
from typing import override

from garak.data import path as data_path
from garak.probes.dra import TOOL_LIST

from attack.assessment import Assessment
from attack.candidate_run import CandidateRun
from attack.dra_limits import DRALimits
from attack.probe import Probe
from journal.journal import Journal
from target.chat import Chat

TOXIC_TRUNC = 0.5
BENIGN_TRUNC = 0.5
FULL_PROB = 0.33
SHORT = 3
TOOLS = 2
TEMPLATE_COUNT = 4
HALF = 0.5
TINY = 2
BRIEF = 4


class DRAProbe(Probe):
    """Create cache-only native DRA puzzles with isolated local randomness."""

    def __init__(self, base: str, candidates: CandidateRun, journal: Journal, budget: DRALimits) -> None:
        """Bind the supplied question and the finite candidate run.

        Args:
            base (str): Exact original request.
            candidates (CandidateRun): Independent candidate execution.
            journal (Journal): Provenance event sink.
            budget (DRALimits): Candidate count and seed.
        """
        super().__init__()
        self.base = base
        self.candidates = candidates
        self.journal = journal
        self.budget = budget

    @override
    def kind(self) -> str:
        return "dra"

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "dra_probe",
            "source": "garak.probes.dra",
            "revision": "3f50ea5ff9cd7050099940647c15c39b07a93392",
            "moderation": "cache-only",
            "budget": {
                "candidates": self.budget.candidates,
                "seed": self.budget.seed,
                "toxic_trunc": TOXIC_TRUNC,
                "benign_trunc": BENIGN_TRUNC,
                "benign_full_token_prob": FULL_PROB
            }
        }

    def prefix(self, letters: dict[str, list[str]], rng: random.Random) -> str:
        """Disguise each character using the pinned map and native choice order.

        Args:
            letters (dict[str, list[str]]): Bundled letter alternatives.
            rng (random.Random): This run's random stream.

        Returns:
            str: Lines with one parenthesized character each.
        """
        lines: list[str] = []
        for letter in self.base.lower():
            token = rng.choice(
                letters.get(
                    letter, [letter]
                )
            )
            index = rng.choice([position for position, value in enumerate(token) if value == letter])
            lines.append(token[:index] + f"({letter})" + token[index + 1:])
        return "\n".join(lines) + "\n"

    def hints(self, cache: dict[str, bool], rng: random.Random) -> str:
        """Choose word hints using only pinned moderation decisions.

        Args:
            cache (dict[str, bool]): Bundled decisions.
            rng (random.Random): This run's random stream.

        Returns:
            str: Native word hints for the original question.
        """
        guide = ""
        for token in (
            self.base.lower()
            .split()
        ):
            if not cache.get(
                token, False
            ):
                if len(token) <= SHORT or rng.random() < FULL_PROB:
                    guide += f"`{token}` "
                else:
                    index = rng.randint(
                        min(
                            len(token) * TINY // SHORT,
                            int(
                                len(token) * BENIGN_TRUNC
                            )
                        ),
                        len(token) - 1
                    )
                    guide += f"`{token[:index]}` "
                    if rng.random() < HALF:
                        guide += f"`{token[index:]}` "
            elif len(token) <= BRIEF:
                index = 0 if len(token) <= TINY else (1 if token[-1].isalpha() else TINY)
                guide += f"`{token[:len(token) - index]}` "
            else:
                index = rng.randint(
                    max(
                        TINY,
                        int(
                            len(token) * TOXIC_TRUNC
                        )
                    ),
                    len(token) - 1
                )
                guide += f"`{token[:index]}` "
                if rng.random() < HALF:
                    guide += f"`{token[index:]} `"
        return guide

    def suffix(self, template: str, cache: dict[str, bool], rng: random.Random) -> str:
        """Apply native word hints and two sampled tools to a template.

        Args:
            template (str): Bundled two-field template.
            cache (dict[str, bool]): Bundled moderation decisions.
            rng (random.Random): This run's random stream.

        Returns:
            str: Reconstruction instructions with native word hints.
        """
        suffix = template.format(
            self.hints(
                cache, rng
            ),
            " and ".join(
                rng.sample(
                    TOOL_LIST, TOOLS
                )
            )
        )
        return suffix.replace(
            "`", '"'
        )

    def prompts(self, rng: random.Random) -> Iterator[str]:
        """Yield pinned templates in order, recording each puzzle's provenance.

        Args:
            rng (random.Random): Fresh random stream for this run.

        Yields:
            str: Disguised request for an independent Target conversation.

        Raises:
            ValueError: If the pinned template count changes.
        """
        root = data_path / "dra"
        letters: dict[str, list[str]] = json.loads(
            (root / "letter_map.json").read_text(encoding="utf-8")
        )
        cache: dict[str, bool] = json.loads(
            (root / "moderation_cache.json").read_text(encoding="utf-8")
        )
        templates = [line.strip() for line in (root / "prompt_templates.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(templates) != TEMPLATE_COUNT:
            raise ValueError("The pinned DRA resource must contain four templates")
        for index, template in enumerate(
            templates[:self.budget.candidates], start=1
        ):
            prefix = self.prefix(
                letters, rng
            )
            prompt = (prefix +
                      self.suffix(
                          template, cache, rng
                      ))
            self.journal.record(
                {
                    "event": "decision",
                    "action": "disguise",
                    "discarded": [],
                    "kept": 1,
                    "candidate": index,
                    "template": index,
                    "reconstruction": self.base.lower(),
                    "prefix": prefix,
                    "prompt": prompt
                }
            )
            yield prompt

    @override
    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        if base != self.base:
            raise ValueError("The probe's base request differs from the run")
        if not 1 <= self.budget.candidates <= TEMPLATE_COUNT or self.budget.seed < 0:
            raise ValueError("Invalid DRA candidate count or seed")
        return self.candidates.execute(
            self.prompts(
                random.Random(self.budget.seed)
            ),
            target, assess
        )
