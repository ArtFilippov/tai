"""Pinned garak DRA methods and resources with an isolated per-run RNG."""

import json
import random
from collections.abc import Iterator
from operator import call
from types import FunctionType
from typing import cast, override

from garak.data import path as data_path
from garak.probes.dra import DRA as NativeDRA

from attacks.shared.contracts import Candidate
from core.primitives.messages import Message
from core.runs.operations import Cause

from .profile import DRAProfile, TEMPLATE_COUNT


def _local(method: FunctionType, rng: random.Random) -> FunctionType:
    globals_copy = {**method.__globals__, "random": rng}
    return FunctionType(method.__code__, globals_copy, method.__name__, method.__defaults__, method.__closure__)


class _LocalDRA(NativeDRA):
    rng = random.Random(0)

    @override
    def _prefix_gen(self, question: str) -> tuple[str, str]:
        return cast(tuple[str, str], call(_local(getattr(NativeDRA, "_prefix_gen"), self.rng), self, question))

    @override
    def _suffix_gen(self, question: str, template: str) -> str:
        return cast(str, call(_local(getattr(NativeDRA, "_suffix_gen"), self.rng), self, question, template))

    def generate(self) -> list[str]:
        return cast(list[str], call(_local(getattr(NativeDRA, "_generate_prompts"), self.rng), self))

    def candidates(self, base: Message) -> Iterator[Candidate]:
        for index, text in enumerate(self.generate()):
            request = base.spawn(text)
            yield Candidate(request, Cause("dra_variant", request.id, metadata={"variant_index": index}))


def candidates(base: Message, *, profile: DRAProfile) -> Iterator[Candidate]:
    # DRA.__init__ seeds global random and loads the bundled harmful-behavior corpus.
    # Populate only the native generation inputs for this one explicit base.
    native = object.__new__(_LocalDRA)
    native.rng = random.Random(profile.seed)
    with open(data_path / "dra" / "moderation_cache.json", encoding="utf-8") as resource:
        native.moderation_cache = json.load(resource)
    with open(data_path / "dra" / "letter_map.json", encoding="utf-8") as resource:
        native.letter_map = json.load(resource)
    with open(data_path / "dra" / "prompt_templates.txt", encoding="utf-8") as resource:
        native.all_templates = [line.strip() for line in resource if line.strip()]
    if len(native.all_templates) != TEMPLATE_COUNT:
        raise ValueError("Pinned garak DRA must contain four templates")
    for name, value in {
        "all_behaviors": [base.to_string()], "toxic_trunc": profile.toxic_trunc, "benign_trunc": profile.benign_trunc,
        "benign_full_token_prob": profile.benign_full_token_prob, "follow_prompt_cap": True,
        "soft_probe_prompt_cap": profile.max_variants,
    }.items():
        setattr(native, name, value)

    yield from call(native.candidates, base)
