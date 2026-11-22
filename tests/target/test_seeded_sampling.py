"""Failure-safe native randomness at the stochastic decoding boundary."""

import unittest

import pytest
import torch

from target.seeded_sampling import SeededSampling


class TestSeededSampling(unittest.TestCase):
    """Restore the caller's RNG even when native sampling raises an error."""

    def test_native_sampling_failure_does_not_leak_random_state(self) -> None:
        state = (
            torch.random.get_rng_state()
            .clone()
        )
        sampling = SeededSampling()
        with pytest.raises(RuntimeError):
            with sampling.scope(
                torch.device("cpu")
            ):
                _ = torch.rand(4)
                _ = torch.multinomial(
                    torch.zeros(4),
                    1
                )
        assert torch.equal(
            state,
            torch.random.get_rng_state()
        ), "A failed sampled operation changed process randomness"
