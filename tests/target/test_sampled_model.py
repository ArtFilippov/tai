"""Real local sampling through the public Model boundary."""

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from target.sampled_model import SampledModel
from target.seeded_sampling import SeededSampling
from .qwen_fixture_utils import model


class TestSampledModel(unittest.TestCase):
    """Advance an isolated random stream without changing shared greedy generation."""

    def test_sampling_varies_repeats_and_preserves_process_randomness(self) -> None:
        with TemporaryDirectory() as directory:
            target = model(
                Path(directory),
                tokens=8, context=256
            )
            state = (
                torch.random.get_rng_state()
                .clone()
            )
            attacker = SampledModel(
                target,
                SeededSampling()
            )
            requests = [
                {
                    "role": "user",
                    "content": "word"
                }
            ]
            first = [attacker.generate(requests) for _ in range(6)]
            repeated = SampledModel(
                target,
                SeededSampling()
            )
            second = [repeated.generate(requests) for _ in range(6)]
            assert first == second and len(set(first)) > 1 and torch.equal(state, torch.random.get_rng_state()), "Sampling repeated one candidate, lost its seed or changed process randomness"

    def test_sampling_does_not_change_greedy_output_or_generation_settings(self) -> None:
        with TemporaryDirectory() as directory:
            target = model(
                Path(directory),
                tokens=8, context=256
            )
            requests = [
                {
                    "role": "user",
                    "content": "word"
                }
            ]
            settings = target.settings()
            original = target.generate(requests)
            _ = SampledModel(
                target,
                SeededSampling()
            ).generate(requests)
            assert (
                target.settings(),
                target.generate(requests)
            ) == (settings, original), "Attacker sampling changed the shared Target decoding policy"

    def test_sampling_metadata_preserves_inherited_backend_generation_options(self) -> None:
        with TemporaryDirectory() as directory:
            target = model(
                Path(directory),
                tokens=8, context=256, repetition=1.1
            )
            attacker = SampledModel(
                target,
                SeededSampling()
            )
            settings = json.loads(
                json.dumps(
                    attacker.settings()
                )
            )["generation"]
            assert (settings["repetition_penalty"], settings["max_new_tokens"], settings["do_sample"]) == (1.1, 150, True), "Sampled metadata omitted the effective inherited repetition penalty"
