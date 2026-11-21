# SPDX-FileCopyrightText: Portions Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite request-bound Doctor candidates assessed in independent conversations."""

from collections.abc import Iterator, Mapping, Sequence
from importlib.resources import files
from random import Random
from typing import override

from garak.attempt import Message

from attack.assessment import Assessment
from attack.candidate_run import CandidateRun
from attack.probe import Probe
from base.base_request import BaseRequest
from journal.journal import Journal
from target.chat import Chat


class DoctorProbe(Probe):
    """Present selected static techniques without sharing Target history or random state."""

    def __init__(self, base: BaseRequest, candidates: CandidateRun, journal: Journal, *,
                 techniques: Sequence[str] = ("puppetry", "bypass", "bypass_leet"), seed: int = 0) -> None:
        """Bind one base request, independent exchanges and reproducible selection.

        Args:
            base (BaseRequest): Original request and provenance.
            candidates (CandidateRun): Independently assessed candidate execution.
            journal (Journal): Run-local transformation and outcome sink.
            techniques (Sequence[str], default=("puppetry", "bypass", "bypass_leet")): Nonempty technique selection.
            seed (int, default=0): Run-local leetspeak seed.

        Raises:
            ValueError: If selection is empty or contains unsupported techniques.
        """
        super().__init__()
        if not techniques or not set(techniques) <= {"puppetry", "bypass", "bypass_leet"}:
            raise ValueError("Doctor techniques must be a nonempty subset of puppetry, bypass, bypass_leet")
        self.base = base
        self.candidates = candidates
        self.journal = journal
        self.techniques: Sequence[str] = tuple(techniques)
        self.seed = seed

    @override
    def kind(self) -> str:
        return "doctor"

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "doctor_probe",
            "source": "garak.probes.doctor",
            "revision": "3f50ea5ff9cd7050099940647c15c39b07a93392",
            "techniques": list(
                self.selection()
            ),
            "seed": self.seed,
            "settings": {
                "candidates": {name: 1 if name == "puppetry" else 2 for name in self.selection()},
                "completions": 1,
                "encoding": "whole prompt; upstream character mapping and random > 0.5",
                "assessment": "original request fulfillment; full raw response"
            }
        }

    @override
    def execute(self, base: str, target: Chat, assess: Assessment) -> Mapping[str, str | None]:
        if base != self.base.text():
            raise ValueError("The Doctor probe is bound to another base request")
        identity: dict[str, str | int] = {}

        def prompts() -> Iterator[Message]:
            random = Random(self.seed)
            position = 0
            for technique in self.selection():
                for index, template in enumerate(
                    self.templates(technique),
                    start=1
                ):
                    position += 1
                    identity.clear()
                    identity.update(
                        technique=technique, template=index, position=position
                    )
                    if technique == "bypass_leet":
                        identity["seed"] = self.seed
                    text = self.render(
                        template, base, technique=technique, random=random
                    )
                    self.journal.record(
                        {
                            "event": "decision",
                            "action": "doctor_transformation",
                            "discarded": [],
                            "kept": 1,
                            "candidate": dict(identity),
                            "original": base,
                            "base_metadata": dict(
                                self.base.metadata()
                            ),
                            "prompt": text
                        }
                    )
                    yield Message(
                        text, notes={"candidate": dict(identity)}
                    )

        def assessment(text: str) -> bool:
            verdict = assess(text)
            if verdict:
                self.journal.record(
                    {
                        "event": "decision",
                        "action": "doctor_success",
                        "discarded": [],
                        "kept": 1,
                        "candidate": dict(identity),
                        "target": target.description(),
                        "text": text
                    }
                )
            return verdict

        return self.candidates.execute(
            prompts(),
            target, assessment
        )

    def selection(self) -> Sequence[str]:
        """Return selected techniques once each in canonical order.

        Returns:
            Sequence[str]: Deduplicated execution selection.
        """
        return tuple(name for name in ("puppetry", "bypass", "bypass_leet") if name in self.techniques)

    def templates(self, technique: str) -> list[str]:
        """Read mechanically transferred text without normalizing its formatting.

        Args:
            technique (str): Validated Doctor technique.

        Returns:
            list[str]: Source templates with an explicit request insertion marker.
        """
        names = ["puppetry.txt"] if technique == "puppetry" else ["bypass_1.txt", "bypass_2.txt"]
        return [files("attack.doctor").joinpath(name).read_text(encoding="utf-8") for name in names]

    def render(self, template: str, base: str, *, technique: str, random: Random) -> str:
        """Substitute the exact request and optionally encode the assembled prompt.

        Args:
            template (str): Source template containing the insertion marker.
            base (str): Unchanged original request.
            technique (str): Validated Doctor technique.
            random (Random): Independent seeded stream for this run.

        Returns:
            str: Complete candidate prompt, preserving upstream text and encoding rules.
        """
        if technique == "bypass_leet":
            template = template.replace(
                "{{HARMFUL_BEHAVIOR}}",
                files("attack.doctor").joinpath("leet_suffix.txt")
                .read_text(encoding="utf-8")
            )
        text = template.replace(
            "{{HARMFUL_BEHAVIOR}}", base
        )
        if technique != "bypass_leet":
            return text
        encoded = text.translate(
            str.maketrans(
                "ieaostIEAOST", "134057134057"
            )
        )
        return "".join(encoded[index] if random.random() > 0.5 else character for index, character in enumerate(text))
