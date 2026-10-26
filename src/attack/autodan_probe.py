"""Pinned garak GA/HGA evolution over an explicit base request."""

import random
from collections.abc import Mapping
from typing import override

import numpy as np
from garak.attempt import Conversation, Message, Turn
from garak.resources.autodan.genetic import autodan_ga, autodan_hga
from garak.resources.autodan.model_utils import check_for_attack_success
from garak.resources.common import REJECTION_STRINGS

from attack.assessment import Assessment
from attack.autodan_corpus import AutoDANCorpus
from attack.autodan_limits import AutoDANLimits
from attack.autodan_mutation_contract import AutoDANMutationContract
from attack.probe import Probe
from journal.journal import Journal
from target.chat import Chat
from target.logits import Logits

MUTATION_ABORT = "AutoDAN mutation failed"
PAIR = 2
GA = "ga"
HGA = "hga"


class AutoDANProbe(Probe):
    """Search by native GA/HGA fitness and check prefix-only replies."""

    def __init__(self, logits: Logits, journal: Journal, *,
                 budget: AutoDANLimits, mutation: AutoDANMutationContract,
                 corpus: AutoDANCorpus) -> None:
        """Bind fitness, mutation and reproducible search limits.

        Args:
            logits (Logits): Independent local fitness operation.
            journal (Journal): Run events.
            budget (AutoDANLimits): Explicit finite search limits.
            mutation (AutoDANMutationContract): Separate native mutation role.
            corpus (AutoDANCorpus): Bundled reference prompts.
        """
        super().__init__()
        self.logits = logits
        self.journal = journal
        self.budget = budget
        self.mutation = mutation
        self.corpus = corpus

    @override
    def kind(self) -> str:
        return f"autodan_{self.budget.mode}"

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": f"autodan_{self.budget.mode}_probe",
            "mode": self.budget.mode,
            "source": "garak.resources.autodan",
            "references": self.corpus.description(),
            "fitness": "prefix + base / target completion",
            "success": "prefix-only / native rejection strings",
            "scorer": self.logits.profile(),
            "attacker": self.mutation.description(),
            "budget": {
                "steps": self.budget.steps,
                "batch": self.budget.batch,
                "elites": self.budget.elites,
                "points": self.budget.points,
                "crossover": self.budget.crossover,
                "mutation": self.budget.mutation,
                "seed": self.budget.seed,
                "target": self.budget.target
            }
        }

    def selected(self, prefixes: list[str], scores: list[float], step: int) -> str:
        """Record native argmin fitness over prefix plus base.

        Args:
            prefixes (list[str]): Scored generation.
            scores (list[float]): Native completion losses.
            step (int): One-based generation number.

        Returns:
            str: Lowest-loss prefix.
        """
        choice = int(
            np.argmin(scores)
        )
        self.journal.record(
            {
                "event": "decision",
                "action": "fitness",
                "discarded": [],
                "kept": len(prefixes),
                "step": step,
                "prefixes": list(prefixes),
                "scores": scores,
                "selected": choice
            }
        )
        return prefixes[choice]

    def evolve(self, prefixes: list[str], scores: list[float], step: int,
               words: dict[str, float]) -> list[str]:
        """Advance the chosen native transition and journal its population.

        Args:
            prefixes (list[str]): Current generation.
            scores (list[float]): Current native losses.
            step (int): One-based generation number.
            words (dict[str, float]): Momentum accumulated within this attempt.

        Returns:
            list[str]: Elites and mutated offspring.

        Raises:
            RuntimeError: If the native mutation fails.
            SystemExit: If the host process exits for another reason.
        """
        try:
            offspring = self.generation(
                prefixes, scores, words
            )
        except SystemExit as exc:
            if exc.code != MUTATION_ABORT:
                raise
            raise RuntimeError("AutoDAN mutation failed") from exc
        self.journal.record(
            {
                "event": "decision",
                "action": "evolve",
                "discarded": [],
                "kept": len(offspring),
                "step": step,
                "prefixes": list(offspring),
                "momentum": dict(words) if self.budget.mode == HGA else {}
            }
        )
        return offspring

    def generation(self, prefixes: list[str], scores: list[float],
                   words: dict[str, float]) -> list[str]:
        """Apply the chosen pinned native operation and retain HGA momentum.

        Args:
            prefixes (list[str]): Current population.
            scores (list[float]): Completion losses.
            words (dict[str, float]): Current word-level momentum.

        Returns:
            list[str]: Next native population.
        """
        if self.budget.mode == HGA:
            offspring, momentum = autodan_hga(
                word_dict=words, control_prefixes=prefixes, score_list=scores,
                num_elites=self.budget.elites, batch_size=self.budget.batch,
                crossover_rate=self.budget.crossover,
                mutation_rate=self.budget.mutation, mutation_generator=self.mutation
            )
            words.clear()
            words.update(momentum)
            return offspring
        return autodan_ga(
            control_prefixes=prefixes, score_list=scores,
            num_elites=self.budget.elites, batch_size=self.budget.batch,
            crossover_rate=self.budget.crossover, num_points=self.budget.points,
            mutation=self.budget.mutation, mutation_generator=self.mutation
        )

    def preflight(self) -> None:
        """Validate native GA/HGA limits before the first model operation.

        Raises:
            ValueError: If the requested search budget is invalid.
        """
        if self.budget.mode not in {GA, HGA}:
            raise ValueError("Invalid AutoDAN mode")
        if self.budget.steps < 1 or not PAIR <= self.budget.batch <= 128:
            raise ValueError("Invalid AutoDAN steps or reference population")
        if not 0 < self.budget.elites < self.budget.batch or (self.budget.mode == GA and (self.budget.batch - self.budget.elites) % PAIR):
            raise ValueError("Invalid AutoDAN elite or offspring count")
        if self.budget.points < 1 or self.budget.seed < 0 or not self.budget.target:
            raise ValueError("Invalid AutoDAN crossover, seed or completion")
        if not 0 <= self.budget.crossover <= 1 or not 0 <= self.budget.mutation <= 1:
            raise ValueError("Invalid AutoDAN probabilities")

    @override
    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        self.preflight()
        random.seed(self.budget.seed)
        np.random.seed(self.budget.seed)
        prefixes = self.corpus.population(self.budget.batch)
        words: dict[str, float] = {}
        for step in range(self.budget.steps):
            scores = [
                self.logits.score(prefix, base, self.budget.target)
                for prefix in prefixes
            ]
            if not all(np.isfinite(score) for score in scores):
                raise ValueError("AutoDAN fitness must be finite")
            prefix = self.selected(
                prefixes, scores, step + 1
            )
            response = target.reply(
                Conversation(
                    [
                        Turn(
                            "user",
                            Message(prefix)
                        )
                    ]
                )
            )
            self.journal.record(
                {
                    "event": "call",
                    "role": "target",
                    "prompt": prefix,
                    "response": response,
                    "step": step + 1
                }
            )
            self.journal.record(
                {
                    "event": "decision",
                    "action": "success" if response and check_for_attack_success(response, REJECTION_STRINGS) else "continue",
                    "discarded": [],
                    "kept": len(prefixes),
                    "step": step + 1,
                    "prefix": prefix,
                    "response": response
                }
            )
            if response and check_for_attack_success(response, REJECTION_STRINGS):
                _ = assess(response)
                return {
                    "text": response,
                    "status": "completed",
                    "reason": "native_success"
                }
            if step + 1 < self.budget.steps:
                prefixes = self.evolve(
                    prefixes, scores, step + 1, words
                )
        return {
            "text": None,
            "status": "no_result",
            "reason": "search_exhausted"
        }
