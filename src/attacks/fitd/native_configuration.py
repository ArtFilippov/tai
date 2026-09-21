"""Run-local configuration of the native FITD plugin."""

from copy import deepcopy
from dataclasses import asdict

from .native import FITD, Prefixes
from .profile import FITDProfile


def configure(probe: FITD, profile: FITDProfile, config_root: object) -> None:
    if config_root not in ({}, None):
        raise ValueError("FITD only supports its explicit run-local configuration")
    for key, value in probe.DEFAULT_PARAMS.items():
        setattr(probe, key, deepcopy(value))
    for key, value in asdict(profile).items():
        setattr(probe, key, value)
    probe.initial_prompt_count, probe.follow_prompt_cap = 1, False
    probe.parallel_attempts, probe.system_prompt = 1, None


def prepare_goal(probe: FITD, goal: str) -> None:
    probe.refusal_detector = Prefixes(config_root={})
    probe.target_prompts = [goal]


def prepare_ladder(probe: FITD, goal: str, queries: list[str]) -> None:
    probe.level_queries_per_goal = {goal: queries}
