"""Research-compatible stochastic decoding with independent seed-zero streams."""

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from random import Random
from typing import override

import torch

from target.sampling import Sampling


class SeededSampling(Sampling):
    """Keep native RNG snapshots and remote request seeds local to one run."""

    def __init__(self) -> None:
        """Create independent streams without touching process randomness."""
        super().__init__()
        self.states: dict[torch.device, torch.Tensor] = {}
        self.random: Random = Random(0)

    @override
    def options(self) -> dict[str, int | float | bool]:
        return {
            "max_new_tokens": 150,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0
        }

    @override
    def scope(self, device: torch.device) -> AbstractContextManager[None]:
        """Advance native randomness while restoring the caller's RNG snapshots.

        Args:
            device (torch.device): Actual inference device.

        Returns:
            AbstractContextManager[None]: Isolated stochastic generation context.
        """
        @contextmanager
        def session() -> Generator[None]:
            process = {torch.device("cpu"): torch.random.get_rng_state()}
            if device.type == "cuda":
                process[device] = torch.cuda.get_rng_state(device)
            if device not in self.states:
                self.states[device] = (
                    torch.Generator(device=device)
                    .manual_seed(0)
                    .get_state()
                )
            try:
                if device.type == "cuda":
                    torch.cuda.set_rng_state(
                        self.states[device], device
                    )
                else:
                    torch.random.set_rng_state(self.states[device])
                yield
            finally:
                self.states[device] = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.random.get_rng_state()
                torch.random.set_rng_state(
                    process[torch.device("cpu")]
                )
                if device.type == "cuda":
                    torch.cuda.set_rng_state(
                        process[device], device
                    )
        return session()

    @override
    def draw(self) -> int:
        return self.random.randrange(2**31)

    @override
    def description(self) -> dict[str, object]:
        return {
            **self.options(),
            "seed": 0,
            "randomness": "advancing run-local stream"
        }
